#include "DQMOffline/Trigger/interface/EgHLTOfflineClient.h"


#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"



#include <boost/algorithm/string.hpp>

EgHLTOfflineClient::EgHLTOfflineClient(const edm::ParameterSet& iConfig)
{
  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogInfo("EgHLTOfflineClient") << "unable to get DQMStore service?";
  }
  if(iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    dbe_->setVerbose(0);
  }
 

  eleHLTFilterNames_ = iConfig.getParameter<std::vector<std::string> >("eleHLTFilterNames");
  eleTightLooseTrigNames_ = iConfig.getParameter<std::vector<std::string> >("eleTightLooseTrigNames");
  phoHLTFilterNames_ = iConfig.getParameter<std::vector<std::string> >("phoHLTFilterNames");
  phoTightLooseTrigNames_ = iConfig.getParameter<std::vector<std::string> >("phoTightLooseTrigNames");

  eleEffVars_=iConfig.getParameter<std::vector<std::string> >("eleEffVars");
  phoEffVars_=iConfig.getParameter<std::vector<std::string> >("phoEffVars");
  eleTrigTPEffVsVars_=iConfig.getParameter<std::vector<std::string> >("eleTrigTPEffVsVars");
  phoTrigTPEffVsVars_=iConfig.getParameter<std::vector<std::string> >("phoTrigTPEffVsVars");
  eleLooseTightTrigEffVsVars_=iConfig.getParameter<std::vector<std::string> >("eleLooseTightTrigEffVsVars");
  phoLooseTightTrigEffVsVars_=iConfig.getParameter<std::vector<std::string> >("phoLooseTightTrigEffVsVars");
 
  dirName_=iConfig.getParameter<std::string>("DQMDirName");
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
  runClient_();
}

//dummy analysis function
void EgHLTOfflineClient::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  
}

void EgHLTOfflineClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c)
{ 
  runClient_();
}

void EgHLTOfflineClient::runClient_()
{
  if(dbe_) dbe_->setCurrentFolder(dirName_);
  //edm::LogInfo("EgHLTOfflineClient") << "end lumi block called";
  std::vector<std::string> regions;
  regions.push_back("eb");
  regions.push_back("ee");
  for(size_t filterNr=0;filterNr<eleHLTFilterNames_.size();filterNr++){
    for(size_t regionNr=0;regionNr<regions.size();regionNr++){
      createTrigTagProbeEffHists(eleHLTFilterNames_[filterNr],regions[regionNr],eleTrigTPEffVsVars_,"gsfEle");
      createN1EffHists(eleHLTFilterNames_[filterNr]+"_gsfEle_effVsEt_tagProbe",regions[regionNr],eleEffVars_);
      createN1EffHists(eleHLTFilterNames_[filterNr]+"_gsfEle_effVsEt",regions[regionNr],eleEffVars_);
      createN1EffHists(eleHLTFilterNames_[filterNr]+"_gsfEle_effVsEt_fakeRate",regions[regionNr],eleEffVars_);
    }
  }
  for(size_t filterNr=0;filterNr<phoHLTFilterNames_.size();filterNr++){
    for(size_t regionNr=0;regionNr<regions.size();regionNr++){
      createN1EffHists(phoHLTFilterNames_[filterNr]+"_pho_effVsEt",regions[regionNr],phoEffVars_);
    }
  }


  for(size_t regionNr=0;regionNr<regions.size();regionNr++){
    createLooseTightTrigEff(eleTightLooseTrigNames_,regions[regionNr],eleLooseTightTrigEffVsVars_,"gsfEle");
    createLooseTightTrigEff(phoTightLooseTrigNames_,regions[regionNr],phoLooseTightTrigEffVsVars_,"pho");
  }
   
  
}

void EgHLTOfflineClient::createN1EffHists(const std::string& baseName,const std::string& region,const std::vector<std::string>& varNames)
{ 
  MonitorElement* numer = dbe_->get(dirName_+"/"+baseName+"_allCuts_"+region);
  
  for(size_t varNr=0;varNr<varNames.size();varNr++){
    MonitorElement* denom = dbe_->get(dirName_+"/"+baseName+"_n1_"+varNames[varNr]+"_"+region);
    if(numer!=NULL && denom!=NULL){
      std::string effHistName(baseName+"_n1Eff_"+varNames[varNr]+"_"+region);
      makeEffMonElemFromPassAndAll(effHistName,numer,denom);   
    }
  }//end loop over varNames 
}

void EgHLTOfflineClient::createTrigTagProbeEffHists(const std::string& filterName,const std::string& region,const std::vector<std::string>& vsVarNames,const std::string& objName)
{
  for(size_t varNr=0;varNr<vsVarNames.size();varNr++){
    std::string allName(dirName_+"/"+filterName+"_trigTagProbe_"+objName+"_all_"+vsVarNames[varNr]+"_"+region);
    MonitorElement* all = dbe_->get(allName); 
    if(all==NULL){
      edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<allName;
      continue;
    }
    std::string passName(dirName_+"/"+filterName+"_trigTagProbe_"+objName+"_pass_"+vsVarNames[varNr]+"_"+region);
    MonitorElement* pass = dbe_->get(passName); 
    if(pass==NULL){
      edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<passName;
      continue;
    }
    
    makeEffMonElemFromPassAndAll(filterName+"_trigTagProbeEff_"+objName+"_vs_"+vsVarNames[varNr]+"_"+region,pass,all);
  }//end loop over vsVarNames
}



void EgHLTOfflineClient::createLooseTightTrigEff(const std::vector<std::string>&  tightLooseTrigNames,const std::string& region,const std::vector<std::string>& vsVarNames,const std::string& objName)
{
  for(size_t varNr=0;varNr<vsVarNames.size();varNr++){
    for(size_t trigNr=0;trigNr<tightLooseTrigNames.size();trigNr++){
      std::vector<std::string> splitString;
      boost::split(splitString,tightLooseTrigNames[trigNr],boost::is_any_of(":"));
      if(splitString.size()!=2) continue; //format incorrect
      const std::string& tightTrig = splitString[0];
      const std::string& looseTrig = splitString[1];
      MonitorElement* fail = dbe_->get(dirName_+"/"+tightTrig+"_"+looseTrig+"_"+objName+"_failTrig_"+vsVarNames[varNr]+"_"+region); 
      if(fail==NULL){
	edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<dirName_+"/"+tightTrig+"_"+looseTrig+"_"+objName+"_failTrig_"+vsVarNames[varNr]+"_"+region;
	continue;
      }

      MonitorElement* pass = dbe_->get(dirName_+"/"+tightTrig+"_"+looseTrig+"_"+objName+"_passTrig_"+vsVarNames[varNr]+"_"+region); 
      if(pass==NULL){
	edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<dirName_+"/"+tightTrig+"_"+looseTrig+"_"+objName+"_passTrig_"+vsVarNames[varNr]+"_"+region;
	continue;
      } 
      const std::string newHistName(tightTrig+"_trigEffTo_"+looseTrig+"_"+objName+"_vs_"+vsVarNames[varNr]+"_"+region);
      makeEffMonElemFromPassAndFail(newHistName,pass,fail);
    }//end loop over trigger pairs
  } //end loop over vsVarNames
  
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
