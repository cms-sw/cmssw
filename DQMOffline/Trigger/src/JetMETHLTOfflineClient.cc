#include "DQMOffline/Trigger/interface/JetMETHLTOfflineClient.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

JetMETHLTOfflineClient::JetMETHLTOfflineClient(const edm::ParameterSet& iConfig):conf_(iConfig)
{
  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogError("JetMETHLTOfflineClient") << "unable to get DQMStore service, upshot is no client histograms will be made";
  }
  if(iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    if(dbe_) dbe_->setVerbose(0);
  }
 
  debug_ = false;
  verbose_ = false;

  processname_ = iConfig.getParameter<std::string>("processname");

  hltTag_ = iConfig.getParameter<std::string>("hltTag");
  if (debug_) std::cout << hltTag_ << std::endl;
  
  dirName_=iConfig.getParameter<std::string>("DQMDirName");
  if(dbe_) dbe_->setCurrentFolder(dirName_);
 
}


JetMETHLTOfflineClient::~JetMETHLTOfflineClient()
{ 
  
}

void JetMETHLTOfflineClient::beginJob(const edm::EventSetup& iSetup)
{
 

}

void JetMETHLTOfflineClient::endJob() 
{

}

void JetMETHLTOfflineClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
 
}


void JetMETHLTOfflineClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  runClient_();
}

//dummy analysis function
void JetMETHLTOfflineClient::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  
}

void JetMETHLTOfflineClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c)
{ 
  runClient_();
}

void JetMETHLTOfflineClient::runClient_()
{
  if(!dbe_) return; //we dont have the DQMStore so we cant do anything
  dbe_->setCurrentFolder(dirName_);

  LogDebug("JetMETHLTOfflineClient") << "runClient" << std::endl;
  if (debug_) std::cout << "runClient" << std::endl; 

  std::vector<MonitorElement*> hltMEs;

  // Look at all folders, go to the subfolder which includes the string "Eff"
  std::vector<std::string> fullPathHLTFolders = dbe_->getSubdirs();
  for(unsigned int i=0;i<fullPathHLTFolders.size();i++) {

    // Move on only if the folder name contains "Eff"
    if (debug_) std::cout << fullPathHLTFolders[i] << std::endl;
    if (fullPathHLTFolders[i].find("Eff")!=string::npos) {
      dbe_->setCurrentFolder(fullPathHLTFolders[i]);
    } else {
      continue;
    }

    // Look at all subfolders, go to the subfolder which includes the string "Eff"
    std::vector<std::string> fullSubPathHLTFolders = dbe_->getSubdirs();
    for(unsigned int j=0;j<fullSubPathHLTFolders.size();j++) {

      if (debug_) std::cout << fullSubPathHLTFolders[j] << std::endl;      
      dbe_->setCurrentFolder(fullSubPathHLTFolders[j]);
    
      // Look at all MonitorElements in this folder
      hltMEs = dbe_->getContents(fullSubPathHLTFolders[j]);
      LogDebug("JetMETHLTOfflineClient")<< "Number of MEs for this HLT path = " << hltMEs.size() << endl;
      
      for(unsigned int k=0;k<hltMEs.size();k++) {
	if (debug_) std::cout << hltMEs[k]->getName() << std::endl;
	if (debug_) std::cout << hltMEs[k]->getTitle() << std::endl;

	//-----
	if (hltMEs[k]->getName().find("Numerator")!=string::npos &&
	    hltMEs[k]->getName().find("Numerator")==0){

	  std::string name = hltMEs[k]->getName();
	  name.erase(0,9); // Removed "Numerator"
	  if( name.find("EtaPhi") !=string::npos ) continue; // do not consider EtaPhi 2D plots

	  MonitorElement* eff ;

	  for(unsigned int l=0;l<hltMEs.size();l++) {
	    if (hltMEs[l]->getName() == "Denominator"+name){
	      // found denominator too
	      
	      bool foundEff=false;
	      for(unsigned int m=0;m<hltMEs.size();m++) {
		if (hltMEs[m]->getName() == "Eff_"+name){
		  foundEff=true;
		  eff = hltMEs[m];
		  break;
		}
	      }

	      TH1F* tNumerator   = hltMEs[k]->getTH1F();
	      TH1F* tDenominator = hltMEs[l]->getTH1F();

	      std::string title = "Eff_"+hltMEs[k]->getTitle();
	      TH1F* teff = (TH1F*) tNumerator->Clone(title.c_str());
	      teff->Divide(tNumerator,tDenominator,1,1,"B");

	      if (foundEff){
// 		*eff->getTH1F()=*teff;
		eff= dbe_->book1D("Eff_"+name,teff);
	      } else {
		eff= dbe_->book1D("Eff_"+name,teff);
	      }
		
	    } // Denominator
	  }   // Loop-l
	}     // Numerator

	//-----
	if (hltMEs[k]->getName().find("EmulatedNumerator")!=string::npos &&
	    hltMEs[k]->getName().find("EmulatedNumerator")==0){

	  std::string name = hltMEs[k]->getName();
	  name.erase(0,17); // Removed "Numerator"
	  if( name.find("EtaPhi") !=string::npos ) continue; // do not consider EtaPhi 2D plots

	  MonitorElement* eff ;

	  for(unsigned int l=0;l<hltMEs.size();l++) {
	    if (hltMEs[l]->getName() == "Denominator"+name){
	      // found denominator too
	      
	      bool foundEff=false;
	      for(unsigned int m=0;m<hltMEs.size();m++) {
		if (hltMEs[m]->getName() == "EmulatedEff_"+name){
		  foundEff=true;
		  eff = hltMEs[m];
		  break;
		}
	      }

	      TH1F* tNumerator   = hltMEs[k]->getTH1F();
	      TH1F* tDenominator = hltMEs[l]->getTH1F();

	      std::string title = "EmulatedEff_"+hltMEs[k]->getTitle();
	      TH1F* teff = (TH1F*) tNumerator->Clone(title.c_str());
	      teff->Divide(tNumerator,tDenominator,1,1,"B");

	      if (foundEff){
// 		*eff->getTH1F()=*teff;
		eff= dbe_->book1D("EmulatedEff_"+name,teff);
	      } else {
		eff= dbe_->book1D("EmulatedEff_"+name,teff);
	      }
		
	    } // Denominator
	  }   // Loop-l
	}     // Numerator

      }       // Loop-k
    }         // fullSubPath
  }           // fullPath

}
