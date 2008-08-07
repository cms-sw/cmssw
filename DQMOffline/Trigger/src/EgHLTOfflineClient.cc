#include "DQMOffline/Trigger/interface/EgHLTOfflineClient.h"

#include "DQMOffline/Trigger/interface/EleHLTPathMon.h"
#include "DQMOffline/Trigger/interface/EleHLTFilterMon.h"
#include "DQMOffline/Trigger/interface/EgHLTOffData.h"


#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "PhysicsTools/Utilities/interface/deltaR.h"

EgHLTOfflineClient::EgHLTOfflineClient(const edm::ParameterSet& iConfig)
{
  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogInfo("EgHLTOfflineClient") << "unable to get DQMStore service?";
  }
  if(iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    dbe_->setVerbose(0);
  }
 
  
  dirName_=iConfig.getParameter<std::string>("DQMDirName");//"HLT/EgHLTOfflineClient_" + iConfig.getParameter<std::string>("sourceModuleName");

  if(dbe_) dbe_->setCurrentFolder(dirName_);
 
}


EgHLTOfflineClient::~EgHLTOfflineClient()
{ 
  
}

void EgHLTOfflineClient::beginJob(const edm::EventSetup& iSetup)
{
 

}

void EgHLTOfflineClient::endJob() 
{
  LogDebug("EgHLTOfflineClient") << "ending job";
}

void EgHLTOfflineClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("EgHLTOfflineClient") << "beginRun, run " << run.id();
}


void EgHLTOfflineClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("EgHLTOfflineClient") << "endRun, run " << run.id();
}

//dummy analysis function
void EgHLTOfflineClient::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
 
}

void EgHLTOfflineClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c)
{
 
  edm::LogInfo("EgHLTOfflineClient") <<" end lumi block called ";

  createEffHist("hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter_gsfEle_effVsEt_n1_dEtaIn");
  createEffHist("hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter_gsfEle_effVsEt_n1_dPhiIn");
  createEffHist("hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter_gsfEle_effVsEt_n1_hOverE");
  createEffHist("hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter_gsfEle_effVsEt_n1_sigmaEtaEta");
}

void EgHLTOfflineClient::createEffHist(const std::string& name)
{
  //edm::LogInfo("EgHLTOfflineClient") <<" getting hist "<<name<<"_all";
  MonitorElement* denom = dbe_->get(dirName_+"/"+name+"_all");
  //edm::LogInfo("EgHLTOfflineClient") <<" got hist "<<denom;
  //edm::LogInfo("EgHLTOfflineClient") <<" getting hist "<<name<<"_pass";
  MonitorElement* numer = dbe_->get(dirName_+"/"+name+"_pass");
  // edm::LogInfo("EgHLTOfflineClient") <<" got hist "<<numer;
  if(denom!=NULL && numer!=NULL){
    TH1F* denomHist = denom->getTH1F();
    TH1F* numerHist = numer->getTH1F();
    
    TH1F* effHist = (TH1F*) numerHist->Clone(name.c_str());
    effHist->Divide(denomHist);
    
    dbe_->book1D(name,effHist);
  }
}


