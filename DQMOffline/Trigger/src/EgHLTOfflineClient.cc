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
 
  eleHLTPathNames_ = iConfig.getParameter<std::vector<std::string> >("eleHLTPathNames");
  eleHLTFilterNames_ = iConfig.getParameter<std::vector<std::string> >("eleHLTFilterNames");
  eleHLTTightLooseFilters_ = iConfig.getParameter<std::vector<std::string> >("eleHLTTightLooseFilters");
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
  edm::LogInfo("EgHLTOfflineClient") << "end lumi block called";
  std::vector<std::string> regions;
  regions.push_back("eb");
  regions.push_back("ee");
  for(size_t pathNr=0;pathNr<eleHLTPathNames_.size();pathNr++){
    for(size_t filterNr=0;filterNr<eleHLTFilterNames_.size();filterNr++){
      for(size_t regionNr=0;regionNr<regions.size();regionNr++){
	createTrigTagProbeEffHists(eleHLTPathNames_[pathNr]+eleHLTFilterNames_[filterNr],regions[regionNr]);
	createN1EffHists(eleHLTPathNames_[pathNr]+eleHLTFilterNames_[filterNr]+"_gsfEle_effVsEt_tagProbe",regions[regionNr]);
	createN1EffHists(eleHLTPathNames_[pathNr]+eleHLTFilterNames_[filterNr]+"_gsfEle_effVsEt",regions[regionNr]);
	createN1EffHists(eleHLTPathNames_[pathNr]+eleHLTFilterNames_[filterNr]+"_gsfEle_effVsEt_fakeRate",regions[regionNr]);
      }
    }
  }

  for(size_t filterNr=0;filterNr<eleHLTTightLooseFilters_.size();filterNr++){
    for(size_t regionNr=0;regionNr<regions.size();regionNr++){
      createLooseTightTrigEff(eleHLTTightLooseFilters_[filterNr],regions[regionNr]);
    }
  }
  
}

void EgHLTOfflineClient::createN1EffHists(const std::string& baseName,const std::string& region)
{ 
  std::vector<std::string> varNames;
  varNames.push_back("dEtaIn");
  varNames.push_back("dPhiIn");
  varNames.push_back("hOverE");
  varNames.push_back("sigmaEtaEta");

  MonitorElement* numer = dbe_->get(dirName_+"/"+baseName+"_allCuts_"+region);
  
  for(size_t varNr=0;varNr<varNames.size();varNr++){
    MonitorElement* denom = dbe_->get(dirName_+"/"+baseName+"_n1_"+varNames[varNr]+"_"+region);
    if(numer!=NULL && denom!=NULL){
      std::string effHistName(baseName+"_n1Eff_"+varNames[varNr]+"_"+region);
      makeEffMonElemFromPassAndAll(effHistName,numer,denom);   
    }
  }//end loop over varNames 
}

void EgHLTOfflineClient::createTrigTagProbeEffHists(const std::string& filterName,const std::string& region)
{
  std::vector<std::string> vsVarNames;
  vsVarNames.push_back("et");
  vsVarNames.push_back("eta");
  vsVarNames.push_back("phi");
  vsVarNames.push_back("charge");

  for(size_t varNr=0;varNr<vsVarNames.size();varNr++){
    std::string allName(dirName_+"/"+filterName+"_trigTagProbe_all_"+vsVarNames[varNr]+"_"+region);
    MonitorElement* all = dbe_->get(allName); 
    if(all==NULL){
      edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<allName;
      continue;
    }
    std::string passName(dirName_+"/"+filterName+"_trigTagProbe_pass_"+vsVarNames[varNr]+"_"+region);
    MonitorElement* pass = dbe_->get(passName); 
    if(pass==NULL){
      edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<passName;
      continue;
    }
    
    makeEffMonElemFromPassAndAll(filterName+"_trigTagProbeEff_vs_"+vsVarNames[varNr]+"_"+region,pass,all);
  }//end loop over vsVarNames
}

MonitorElement* EgHLTOfflineClient::makeEffMonElemFromPassAndAll(const std::string& name,const MonitorElement* pass,const MonitorElement* all)
{
  TH1F* passHist = pass->getTH1F();
  if(passHist->GetSumw2N()==0) passHist->Sumw2();
  TH1F* allHist = all->getTH1F();
  if(allHist->GetSumw2N()==0) allHist->Sumw2();
  TH1F* effHist = (TH1F*) passHist->Clone(name.c_str());
  effHist->Divide(passHist,allHist,1,1,"B");
  
  MonitorElement* eff = dbe_->get(dirName_+"/"+name);
  if(eff==NULL) eff= dbe_->book1D(name,effHist);
  else{ //I was having problems with collating the histograms, hence why I'm just reseting the histogram value
    *eff->getTH1F()=*effHist; 
    delete effHist;
  }
  return eff;
}

MonitorElement* EgHLTOfflineClient::makeEffMonElemFromPassAndFail(const std::string& name,const MonitorElement* pass,const MonitorElement* fail)
{
  TH1F* failHist = fail->getTH1F();   
  if(failHist->GetSumw2N()==0) failHist->Sumw2();
  TH1F* passHist = pass->getTH1F();
  if(passHist->GetSumw2N()==0) passHist->Sumw2();
  
  TH1F* effHist = (TH1F*) passHist->Clone(name.c_str());
  effHist->Add(failHist);
  effHist->Divide(passHist,effHist,1,1,"B");
  
  MonitorElement* eff = dbe_->get(dirName_+"/"+name);
  if(eff==NULL) eff = dbe_->book1D(name,effHist);
  else{ //I was having problems with collating the histograms, hence why I'm just reseting the histogram value
    *eff->getTH1F()=*effHist; 
    delete effHist;
  }
  return eff;
}

void EgHLTOfflineClient::createLooseTightTrigEff(const std::string& filterName,const std::string& region)
{
  std::vector<std::string> vsVarNames;
  vsVarNames.push_back("et");
  vsVarNames.push_back("eta");
  vsVarNames.push_back("charge");

  for(size_t varNr=0;varNr<vsVarNames.size();varNr++){
    MonitorElement* fail = dbe_->get(dirName_+"/"+filterName+"_passLooseTrig_failTightTrig_"+vsVarNames[varNr]+"_"+region); 
    if(fail==NULL){
      edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<dirName_+"/"+filterName+"_passLooseTrig_failTightTrig_"+vsVarNames[varNr]+"_"+region;
      continue;
    }

    MonitorElement* pass = dbe_->get(dirName_+"/"+filterName+"_passLooseTrig_passTightTrig_"+vsVarNames[varNr]+"_"+region); 
    if(pass==NULL){
      edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<dirName_+"/"+filterName+"_passLooseTrig_passTightTrig_"+vsVarNames[varNr]+"_"+region;
      continue;
    } 
    const std::string newHistName(filterName+"_tightTrigEff_vs_"+vsVarNames[varNr]+"_"+region);
    makeEffMonElemFromPassAndFail(newHistName,pass,fail);
  } //end loop over vsVarNames
  
}
  
