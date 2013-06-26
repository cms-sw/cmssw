#include "DQMOffline/Trigger/interface/BTagHLTOfflineClient.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

BTagHLTOfflineClient::BTagHLTOfflineClient(const edm::ParameterSet& iConfig):conf_(iConfig)
{
  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogError("BTagHLTOfflineClient") << "unable to get DQMStore service, upshot is no client histograms will be made";
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


BTagHLTOfflineClient::~BTagHLTOfflineClient()
{ 
  
}

void BTagHLTOfflineClient::beginJob()
{

}

void BTagHLTOfflineClient::endJob() 
{

}

void BTagHLTOfflineClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  
}


void BTagHLTOfflineClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  runClient_();
}

//dummy analysis function
void BTagHLTOfflineClient::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  
}

void BTagHLTOfflineClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c)
{ 
  runClient_();
}

void BTagHLTOfflineClient::runClient_()
{
  if(!dbe_) return; //we dont have the DQMStore so we cant do anything
  dbe_->setCurrentFolder(dirName_);
  
  LogDebug("BTagHLTOfflineClient") << "runClient" << std::endl;
  if (debug_) std::cout << "runClient" << std::endl; 
  
  std::vector<MonitorElement*> hltMEs;
  
  // Look at all folders, go to the subfolder which includes the string "Eff"
  std::vector<std::string> fullPathHLTFolders = dbe_->getSubdirs();
  for(unsigned int i=0;i<fullPathHLTFolders.size();i++) {
    
    // Move on only if the folder name contains "Eff" Or "Trigger Summary"
    if (debug_) std::cout << fullPathHLTFolders[i] << std::endl;
    if ((fullPathHLTFolders[i].find("Eff")!=std::string::npos)) {
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
      LogDebug("BTagHLTOfflineClient")<< "Number of MEs for this HLT path = " << hltMEs.size() << std::endl;
      
      for(unsigned int k=0;k<hltMEs.size();k++) {
	if (debug_) std::cout << hltMEs[k]->getName() << std::endl;

	//-----
	if ((hltMEs[k]->getName().find("ME_Numerator")!=std::string::npos) && hltMEs[k]->getName().find("ME_Numerator")==0){

	  std::string name = hltMEs[k]->getName();
	  name.erase(0,12); // Removed "ME_Numerator"
          if (debug_) std::cout <<"==name=="<< name << std::endl;
	  
	  for(unsigned int l=0;l<hltMEs.size();l++) {
	    if (hltMEs[l]->getName() == "ME_Denominator"+name){
	      // found denominator too
              if(name.find("EtaPhi") !=std::string::npos) 
		{
		  TH2F* tNumerator   = hltMEs[k]->getTH2F();
		  TH2F* tDenominator = hltMEs[l]->getTH2F();
		  
		  std::string title = "Eff_"+hltMEs[k]->getTitle();
		  
		  TH2F *teff = (TH2F*) tNumerator->Clone(title.c_str());
		  teff->Divide(tNumerator,tDenominator,1,1);
		  dbe_->book2D("ME_Eff_"+name,teff);
		  delete teff;			
		}
	      else{
		TH1F* tNumerator   = hltMEs[k]->getTH1F();
		TH1F* tDenominator = hltMEs[l]->getTH1F();
		
		std::string title = "Eff_"+hltMEs[k]->getTitle();
		
		TH1F *teff = (TH1F*) tNumerator->Clone(title.c_str());
		teff->Divide(tNumerator,tDenominator,1,1);
		dbe_->book1D("ME_Eff_"+name,teff);
		delete teff;
	      }
	    } // Denominator
	  }   // Loop-l
	}     // Numerator
	
        
      }       // Loop-k
    }         // fullSubPath
  }           // fullPath
}

