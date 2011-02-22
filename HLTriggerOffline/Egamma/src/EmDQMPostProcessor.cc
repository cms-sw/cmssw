#include "HLTriggerOffline/Egamma/interface/EmDQMPostProcessor.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include <iostream>
#include <string.h>
#include <iomanip>
#include <fstream>
#include <math.h>


EmDQMPostProcessor::EmDQMPostProcessor(const edm::ParameterSet& pset)
{
  subDir_ = pset.getUntrackedParameter<std::string>("subDir");
}

void EmDQMPostProcessor::endRun(edm::Run const& run, edm::EventSetup const& es)
{
  //////////////////////////////////
  // setup DQM stor               //
  //////////////////////////////////
  
  DQMStore * dqm = 0;
  dqm = edm::Service<DQMStore>().operator->();

  if ( ! dqm ) {
    edm::LogInfo("EmDQMPostProcessor") << "Cannot create DQMStore instance\n";
    return;
  }


  //go to the directory to be processed
  if(dqm->dirExists(subDir_)) dqm->cd(subDir_);
  else {
    edm::LogWarning("EmDQMPostProcessor") << "cannot find directory: " << subDir_ << " , skipping";
    return;
  }

  //////////////////////////////////
  //loop over all triggers/samples//
  //////////////////////////////////

  std::vector<std::string> subdirectories = dqm->getSubdirs();
  for(std::vector<std::string>::iterator dir = subdirectories.begin() ;dir!= subdirectories.end(); dir++ ){
    dqm->cd(*dir);


    ////////////////////////////////////////////////////////
    // Do everything twice: once for mc-matched histos,   //
    // once for unmatched histos                          //
    ////////////////////////////////////////////////////////

    std::vector<std::string> postfixes;
    std::string tmpstring=""; //unmatched histos
    postfixes.push_back(tmpstring);
    tmpstring="_MC_matched";
    postfixes.push_back(tmpstring);

    for(std::vector<std::string>::iterator postfix=postfixes.begin(); postfix!=postfixes.end();postfix++){
      
      /////////////////////////////////////
      // computer per-event efficiencies //
      /////////////////////////////////////
      
      std::string histoName="efficiency_by_step"+ *postfix;
      std::string baseName = "total_eff"+ *postfix;
      MonitorElement* total = dqm->book1D(histoName.c_str(),dqm->get(dqm->pwd() + "/" + baseName)->getTH1F());
      total->setTitle(histoName);
      
      //     std::vector<std::string> mes = dqm->getMEs();
      //     for(std::vector<std::string>::iterator me = mes.begin() ;me!= mes.end(); me++ )
      //       std::cout <<*me <<std::endl;
      //     std::cout <<std::endl;
      
      float value=0;
      float error=0;    
      //compute stepwise total efficiencies 
      for(int bin= total->getNbinsX()-2 ; bin > 1  ; bin--){
	value=0;
	error=0;
	if(total->getBinContent(bin-1) != 0){
	  value = total->getBinContent(bin)/total->getBinContent(bin-1) ;
	  error = sqrt(value*(1-value)/total->getBinContent(bin-1));
	}
	total->setBinContent(bin,value);
	total->setBinError(bin,error);
      }
      
      //set first bin to L1 efficiency
      if(total->getBinContent(total->getNbinsX()) !=0 ){
      	value = total->getBinContent(1)/total->getBinContent(total->getNbinsX()) ;
	error = sqrt(value*(1-value)/total->getBinContent(total->getNbinsX()));
      }else{
      value=0;error=0;
      }
      total->setBinContent(1,value);
      total->setBinError(1,error);
      
      //total efficiency relative to gen
      if(total->getBinContent(total->getNbinsX()) !=0 ){
      	value = dqm->get(dqm->pwd() + "/" + baseName)->getBinContent(total->getNbinsX()-2)/total->getBinContent(total->getNbinsX()) ;
	error = sqrt(value*(1-value)/total->getBinContent(total->getNbinsX()));
      }else{
	value=0;error=0;
      }
      total->setBinContent(total->getNbinsX(),value);
      total->setBinError(total->getNbinsX(),error);
      total->setBinLabel(total->getNbinsX(),"total efficiency rel. gen");
      
      //total efficiency relative to L1
      if(total->getBinContent(total->getNbinsX()) !=0 ){
      	value = dqm->get(dqm->pwd() + "/" + baseName)->getBinContent(total->getNbinsX()-2)/dqm->get(dqm->pwd() + "/" + baseName)->getBinContent(1) ;
	error = sqrt(value*(1-value)/dqm->get(dqm->pwd() + "/" + baseName)->getBinContent(1));
      }else{
	value=0;error=0;
      }
      total->setBinContent(total->getNbinsX()-1,value);
      total->setBinError(total->getNbinsX()-1,error);
      total->setBinLabel(total->getNbinsX()-1,"total efficiency rel. L1");
      
      total->getTH1F()->SetMaximum(1.2);
      total->getTH1F()->SetMinimum(0);
      
      ///////////////////////////////////////////
      // compute per-object efficiencies       //
      ///////////////////////////////////////////
      MonitorElement *eff, *num, *denom, *genPlot, *effVsGen, *effL1VsGen;
      std::vector<std::string> varNames; 
      varNames.push_back("eta"); 
      varNames.push_back("phi"); 
      varNames.push_back("et");

      std::string filterName;
      std::string filterName2;
      std::string denomName;
      std::string numName;

      // Get the gen-level plots
      std::string genName;

      // Get the L1 over gen filter first
      filterName2= total->getTH1F()->GetXaxis()->GetBinLabel(1);
	
      //loop over variables (eta/phi/et)
      for(std::vector<std::string>::iterator var = varNames.begin(); var != varNames.end() ; var++){
	numName   = dqm->pwd() + "/" + filterName2 + *var + *postfix;
	genName   = dqm->pwd() + "/gen_" + *var ;
	num       = dqm->get(numName);
	genPlot   = dqm->get(genName);
        effL1VsGen = dqm->book1D("efficiency_"+filterName2+"_vs_"+*var +*postfix, dqm->get(numName)->getTH1F());

	// Check if histograms actually exist
	if(!num || !genPlot) break; 
	
	// Make sure we are able to book new element
	if (!dqm) break;

	// Create the efficiency plot
	effL1VsGen->setTitle("efficiency_"+filterName2+"_vs_"+*var + *postfix);
	effL1VsGen->getTH1F()->SetMaximum(1.2);
	effL1VsGen->getTH1F()->SetMinimum(0);
	effL1VsGen->getTH1F()->GetXaxis()->SetTitle(var->c_str());
	effL1VsGen->getTH1F()->Divide(num->getTH1F(),genPlot->getTH1F(),1,1,"b" );
      }
    
      // get the filter names from the bin-labels of the master-histogram
      for (int filter=1; filter < total->getNbinsX()-2; filter++) {
	filterName = total->getTH1F()->GetXaxis()->GetBinLabel(filter);
	filterName2= total->getTH1F()->GetXaxis()->GetBinLabel(filter+1);
	
	//loop over variables (eta/et)
	for(std::vector<std::string>::iterator var = varNames.begin(); var != varNames.end() ; var++){
	  numName   = dqm->pwd() + "/" + filterName2 + *var + *postfix;
	  denomName = dqm->pwd() + "/" + filterName  + *var + *postfix;
	  num       = dqm->get(numName);
	  denom     = dqm->get(denomName);

          // Check if histograms actually exist
	  if(!num || !denom) break; 

	  // Make sure we are able to book new element
          if (!dqm) break;

	  // Is this the last filter? Book efficiency vs gen level
	  std::string temp = *postfix;
          if (filter==total->getNbinsX()-3 && temp.find("matched")!=std::string::npos) {
            genName = dqm->pwd() + "/gen_" + *var;
            genPlot = dqm->get(genName);
            effVsGen = dqm->book1D("final_eff_vs_"+*var, dqm->get(genName)->getTH1F());
            if (!dqm) break;

            effVsGen->setTitle("Efficiency Compared to Gen vs "+*var);
	    effVsGen->getTH1F()->SetMaximum(1.2); effVsGen->getTH1F()->SetMinimum(0.0);
	    effVsGen->getTH1F()->GetXaxis()->SetTitle(var->c_str());
	    effVsGen->getTH1F()->Divide(num->getTH1F(),genPlot->getTH1F(),1,1,"b" );
          }

	  eff = dqm->book1D("efficiency_"+filterName2+"_vs_"+*var +*postfix, dqm->get(numName)->getTH1F());
	  
          // Make sure we were able to book new element
          if (!dqm) break;

          // Create the efficiency plot
          /* num->getTH1F()->Sumw2();
	  denom->getTH1F()->Sumw2(); 
          eff->getTH1F()->Sumw2(); */

	  eff->setTitle("efficiency_"+filterName2+"_vs_"+*var + *postfix);
	  eff->getTH1F()->SetMaximum(1.2);
	  eff->getTH1F()->SetMinimum(0);
	  eff->getTH1F()->GetXaxis()->SetTitle(var->c_str());
	  eff->getTH1F()->Divide(num->getTH1F(),denom->getTH1F(),1,1,"b" );
	}
      }
    } 
    dqm->goUp();
  }
  
}
DEFINE_FWK_MODULE(EmDQMPostProcessor);
