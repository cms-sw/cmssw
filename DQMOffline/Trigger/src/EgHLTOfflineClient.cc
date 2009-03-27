#include "DQMOffline/Trigger/interface/EgHLTOfflineClient.h"


#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"



#include <boost/algorithm/string.hpp>

EgHLTOfflineClient::EgHLTOfflineClient(const edm::ParameterSet& iConfig):dbe_(NULL)
{
  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogError("EgHLTOfflineClient") << "unable to get DQMStore service, upshot is no client histograms will be made";
  }
  if(iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    if(dbe_) dbe_->setVerbose(0);
  }
 

  eleHLTFilterNames_ = iConfig.getParameter<std::vector<std::string> >("eleHLTFilterNames");
  eleTightLooseTrigNames_ = iConfig.getParameter<std::vector<std::string> >("eleTightLooseTrigNames");
  phoHLTFilterNames_ = iConfig.getParameter<std::vector<std::string> >("phoHLTFilterNames");
  phoTightLooseTrigNames_ = iConfig.getParameter<std::vector<std::string> >("phoTightLooseTrigNames");

  eleN1EffVars_=iConfig.getParameter<std::vector<std::string> >("eleN1EffVars");
  eleSingleEffVars_ = iConfig.getParameter<std::vector<std::string> >("eleSingleEffVars");
  eleEffTags_ = iConfig.getParameter<std::vector<std::string> >("eleEffTags");
  eleTrigTPEffVsVars_ = iConfig.getParameter<std::vector<std::string> >("eleTrigTPEffVsVars");
  eleLooseTightTrigEffVsVars_ =  iConfig.getParameter<std::vector<std::string> >("eleLooseTightTrigEffVsVars");

  phoN1EffVars_=iConfig.getParameter<std::vector<std::string> >("phoN1EffVars");
  phoSingleEffVars_ = iConfig.getParameter<std::vector<std::string> >("phoSingleEffVars");
  phoEffTags_ = iConfig.getParameter<std::vector<std::string> >("phoEffTags");
  phoTrigTPEffVsVars_ = iConfig.getParameter<std::vector<std::string> >("phoTrigTPEffVsVars");
  phoLooseTightTrigEffVsVars_ =  iConfig.getParameter<std::vector<std::string> >("phoLooseTightTrigEffVsVars");
  
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

}

void EgHLTOfflineClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
 
}


void EgHLTOfflineClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
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
  if(!dbe_) return; //we dont have the DQMStore so we cant do anything
  dbe_->setCurrentFolder(dirName_);
 

  std::vector<std::string> regions;
  regions.push_back("eb");
  regions.push_back("ee");

  
  for(size_t filterNr=0;filterNr<eleHLTFilterNames_.size();filterNr++){
    for(size_t regionNr=0;regionNr<regions.size();regionNr++){
      for(size_t effNr=0;effNr<eleEffTags_.size();effNr++){
	createN1EffHists(eleHLTFilterNames_[filterNr]+"_gsfEle_"+eleEffTags_[effNr],regions[regionNr],eleN1EffVars_);
	createSingleEffHists(eleHLTFilterNames_[filterNr]+"_gsfEle_"+eleEffTags_[effNr],regions[regionNr],eleSingleEffVars_);
	createTrigTagProbeEffHists(eleHLTFilterNames_[filterNr],regions[regionNr],eleTrigTPEffVsVars_,"gsfEle");
      }
    }
  }
  
  
  for(size_t filterNr=0;filterNr<phoHLTFilterNames_.size();filterNr++){
    for(size_t regionNr=0;regionNr<regions.size();regionNr++){
      for(size_t effNr=0;effNr<phoEffTags_.size();effNr++){
	createN1EffHists(eleHLTFilterNames_[filterNr]+"_pho_"+phoEffTags_[effNr],regions[regionNr],phoN1EffVars_);
	createSingleEffHists(eleHLTFilterNames_[filterNr]+"_gsfEle_"+phoEffTags_[effNr],regions[regionNr],phoSingleEffVars_);
      }
    }
  }

  for(size_t regionNr=0;regionNr<regions.size();regionNr++){
    createLooseTightTrigEff(eleTightLooseTrigNames_,regions[regionNr],eleLooseTightTrigEffVsVars_,"gsfEle");   
    createLooseTightTrigEff(eleTightLooseTrigNames_,regions[regionNr],eleLooseTightTrigEffVsVars_,"gsfEle_trigCuts");
    createLooseTightTrigEff(phoTightLooseTrigNames_,regions[regionNr],phoLooseTightTrigEffVsVars_,"pho"); 
    createLooseTightTrigEff(phoTightLooseTrigNames_,regions[regionNr],phoLooseTightTrigEffVsVars_,"pho_trigCuts");
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

void EgHLTOfflineClient::createSingleEffHists(const std::string& baseName,const std::string& region,const std::vector<std::string>& varNames)
{ 
  MonitorElement* denom = dbe_->get(dirName_+"/"+baseName+"_noCuts_"+region);
  
  for(size_t varNr=0;varNr<varNames.size();varNr++){
    MonitorElement* numer = dbe_->get(dirName_+"/"+baseName+"_single_"+varNames[varNr]+"_"+region);
    if(numer!=NULL && denom!=NULL){
      std::string effHistName(baseName+"_singleEff_"+varNames[varNr]+"_"+region);
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
      //edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<allName;
      continue;
    }
    std::string passName(dirName_+"/"+filterName+"_trigTagProbe_"+objName+"_pass_"+vsVarNames[varNr]+"_"+region);
    MonitorElement* pass = dbe_->get(passName); 
    if(pass==NULL){
      //edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<passName;
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
	//edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<dirName_+"/"+tightTrig+"_"+looseTrig+"_"+objName+"_failTrig_"+vsVarNames[varNr]+"_"+region;

	continue;
      }

      MonitorElement* pass = dbe_->get(dirName_+"/"+tightTrig+"_"+looseTrig+"_"+objName+"_passTrig_"+vsVarNames[varNr]+"_"+region); 
      if(pass==NULL){

	//edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<dirName_+"/"+tightTrig+"_"+looseTrig+"_"+objName+"_passTrig_"+vsVarNames[varNr]+"_"+region;
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
