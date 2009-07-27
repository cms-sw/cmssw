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

void JetMETDQMOfflineClient::beginJob(const edm::EventSetup& iSetup)
{
 

}

void JetMETDQMOfflineClient::endJob() 
{

}

void JetMETDQMOfflineClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
 
}


void JetMETDQMOfflineClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  runClient_();
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
  
  if(!dbe_) return; //we dont have the DQMStore so we cant do anything

  LogDebug("JetMETDQMOfflineClient") << "runClient" << std::endl;
  if (verbose_) std::cout << "runClient" << std::endl; 

//   std::vector<MonitorElement*> headMEs;
  std::vector<MonitorElement*> MEs;

  ////////////////////////////////////////////////////////
  // Total number of lumi sections and time for this run
  ///////////////////////////////////////////////////////

  TH1F* tlumisec;
  MonitorElement *meLumiSec = dbe_->get("JetMET/lumisec");

  int    totlsec=0;
  double totltime=0;

  if ( meLumiSec->getRootObject() ){

    tlumisec = meLumiSec->getTH1F();    
    for (int i=0; i<500; i++){
      if (tlumisec->GetBinContent(i+1)) totlsec++;
    }

  }
  totltime = totlsec * 90.;

  /////////////////////////
  // MET
  /////////////////////////

  dbe_->setCurrentFolder(dirName_+"/"+dirNameMET_);

  MonitorElement *me;
  TH1F *tCaloMET;
  TH1F *tCaloMETRate;
  MonitorElement *hCaloMETRate;

  // Look at all folders, go to the subfolder which includes the string "Eff"
  std::vector<std::string> fullPathDQMFolders = dbe_->getSubdirs();
  for(unsigned int i=0;i<fullPathDQMFolders.size();i++) {

    if (verbose_) std::cout << fullPathDQMFolders[i] << std::endl;      
    dbe_->setCurrentFolder(fullPathDQMFolders[i]);

    // Look at all MonitorElements in this folder
    me = dbe_->get(fullPathDQMFolders[i]+"/"+"METTask_CaloMET");

    if ( me ) {
    if ( me->getRootObject() ) {

      tCaloMET     = me->getTH1F();

      // Integral plot 
      tCaloMETRate = (TH1F*) tCaloMET->Clone("METTask_CaloMETRate2");
      for (int i = tCaloMETRate->GetNbinsX()-1; i>=0; i--){
	tCaloMETRate->SetBinContent(i+1,tCaloMETRate->GetBinContent(i+2)+tCaloMET->GetBinContent(i+1));
      }
      
      // Convert number of events to Rate (Hz)
      if (totltime){
	for (int i=tCaloMETRate->GetNbinsX()-1;i>=0;i--){
	  tCaloMETRate->SetBinContent(i+1,tCaloMETRate->GetBinContent(i+1)/double(totltime));
	}
      }

      hCaloMETRate      = dbe_->book1D("METTask_CaloMETRate2",tCaloMETRate);

    } // me->getRootObject()
    } // me
  }   // fullPathDQMFolders-loop

  /////////////////////////
  // Jet
  /////////////////////////
   


}

