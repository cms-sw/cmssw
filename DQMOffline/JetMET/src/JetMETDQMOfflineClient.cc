#include "DQMOffline/JetMET/interface/JetMETDQMOfflineClient.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

JetMETDQMOfflineClient::JetMETDQMOfflineClient(const edm::ParameterSet& iConfig):conf_(iConfig)
{

  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogError("JetMETDQMOfflineClient") 
    << "unable to get DQMStore service, upshot is no client histograms will be made";
  }
  if(iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    if(dbe_) dbe_->setVerbose(0);
  }
 
  verbose_ = conf_.getUntrackedParameter<int>("Verbose");

  dirName_=iConfig.getUntrackedParameter<std::string>("DQMDirName");
  if(dbe_) dbe_->setCurrentFolder(dirName_);

  dirNameJet_=iConfig.getUntrackedParameter<std::string>("DQMJetDirName");
  dirNameMET_=iConfig.getUntrackedParameter<std::string>("DQMMETDirName");
 
}


JetMETDQMOfflineClient::~JetMETDQMOfflineClient()
{ 
}

void JetMETDQMOfflineClient::beginJob(void)
{
 

}

void JetMETDQMOfflineClient::endJob() 
{
}

// void JetMETDQMOfflineClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
// {
// }



void JetMETDQMOfflineClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  //runClient_();
}

//dummy analysis function
void JetMETDQMOfflineClient::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
}

void JetMETDQMOfflineClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c)
{ 
}

void JetMETDQMOfflineClient::runClient_()
{
  /* -> calculates right now a histogram out of the JID pass fraction with binomial errors, but histogram is later on never used
  if(!dbe_) return; //we dont have the DQMStore so we cant do anything

  LogDebug("JetMETDQMOfflineClient") << "runClient" << std::endl;
  if (verbose_) std::cout << "runClient" << std::endl; 

  std::vector<MonitorElement*> MEs;
  std::vector<std::string> fullPathDQMFolders = dbe_->getSubdirs();

  dbe_->setCurrentFolder(dirName_+"/"+dirNameJet_);

  // Look at all folders (JetMET/Jet/AntiKtJets,JetMET/Jet/CleanedAntiKtJets, etc)
  fullPathDQMFolders.clear();
  fullPathDQMFolders = dbe_->getSubdirs();
  for(unsigned int i=0;i<fullPathDQMFolders.size();i++) {
    if (verbose_) std::cout << fullPathDQMFolders[i] << std::endl;      
    dbe_->setCurrentFolder(fullPathDQMFolders[i]);
    std::vector<std::string> getMEs = dbe_->getMEs();
    std::vector<std::string>::const_iterator cii;
    for(cii=getMEs.begin(); cii!=getMEs.end(); cii++) {
      if ((*cii).find("_binom")!=std::string::npos) continue;
      if ((*cii).find("JIDPassFractionVS")!=std::string::npos){  // Look for MEs with "JIDPassFractionVS"
	me = dbe_->get(fullPathDQMFolders[i]+"/"+(*cii));
	if ( me ) {	  
	  if ( me->getRootObject() ) {
	    TProfile *tpro = (TProfile*) me->getRootObject();
	    TH1F *tPassFraction = new TH1F(((*cii)+"_binom").c_str(),((*cii)+"_binom").c_str(),
					   tpro->GetNbinsX(),tpro->GetBinLowEdge(1),tpro->GetBinLowEdge(tpro->GetNbinsX()+1));
	    for (int ibin=0; ibin<tpro->GetNbinsX(); ibin++){
	      double nentries = tpro->GetBinEntries(ibin+1);
	      double epsilon  = tpro->GetBinContent(ibin+1);
	      if (epsilon>1. || epsilon<0.) continue;
	      tPassFraction->SetBinContent(ibin+1,epsilon);                        // 
	      if(nentries>0) tPassFraction->SetBinError(ibin+1,pow(epsilon*(1.-epsilon)/nentries,0.5));
	    }
	  } // me->getRootObject()
	} // me
      }// if find
    }   // cii-loop
    
  } // i-loop 
  */
}

