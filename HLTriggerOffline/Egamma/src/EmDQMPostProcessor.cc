
#include "HLTriggerOffline/Egamma/interface/EmDQMPostProcessor.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include <iostream>
#include <string.h>
#include <iomanip>
#include<fstream>
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
    // Do ereything twice: once for mc-matched histos,    //
    // once for unmatched histos                          //
    ////////////////////////////////////////////////////////

    std::vector<std::string> postfixes;
    std::string tmpstring=""; //unmatched histos
    postfixes.push_back(tmpstring);
    tmpstring=" MC matched";
    postfixes.push_back(tmpstring);

    for(std::vector<std::string>::iterator postfix=postfixes.begin(); postfix!=postfixes.end();postfix++){
      
      /////////////////////////////////////
      // computer per-event efficiencies //
      /////////////////////////////////////
      
      std::string histoName="efficiency by step"+ *postfix;
      std::string baseName = "total eff"+ *postfix;
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
      MonitorElement *eff,*num,*denom;
      std::vector<std::string> varnames; varnames.push_back("eta"); varnames.push_back("et");
      std::string filtername;      
      std::string filtername2;
      std::string denomname;
      std::string numname;
      // get the filter names from the bin-labels of the master-histogram
      for(int filter=1;filter < total->getNbinsX()-2; filter++){
	filtername = total->getTH1F()->GetXaxis()->GetBinLabel(filter);
	filtername2= total->getTH1F()->GetXaxis()->GetBinLabel(filter+1);
	
	//loop over variables (eta/et)
	for(std::vector<std::string>::iterator var = varnames.begin(); var != varnames.end() ; var++){
	  numname  = dqm->pwd() + "/"+ filtername2 + *var + *postfix;
	  denomname= dqm->pwd() + "/"+ filtername + *var + *postfix;
	  num   =dqm->get(numname);
	  denom =dqm->get(denomname);
	  if(!num || !denom) break; // dont try to devide if the histos aren't there
	  eff = dqm->book1D("efficiency "+filtername2+" vs "+*var +*postfix ,dqm->get(numname)->getTH1F());
	  if(!dqm) break; // couldnt create new element => don't fill it;
	  eff->setTitle("efficiency "+filtername2+"vs "+*var + *postfix);
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
