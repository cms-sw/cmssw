#include "DQMOffline/Trigger/interface/EgHLTOfflineClient.h"


#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMOffline/Trigger/interface/EgHLTTrigTools.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <boost/algorithm/string.hpp>

#include "TGraphAsymmErrors.h"

EgHLTOfflineClient::EgHLTOfflineClient(const edm::ParameterSet& iConfig):isSetup_(false)
{

  eleHLTFilterNames_ = iConfig.getParameter<std::vector<std::string> >("eleHLTFilterNames");
  eleHLTFilterNames2Leg_ = iConfig.getParameter<std::vector<std::string> >("eleHLTFilterNames2Leg");
  eleTightLooseTrigNames_ = iConfig.getParameter<std::vector<std::string> >("eleTightLooseTrigNames");
  phoHLTFilterNames_ = iConfig.getParameter<std::vector<std::string> >("phoHLTFilterNames");
  phoHLTFilterNames2Leg_ = iConfig.getParameter<std::vector<std::string> >("phoHLTFilterNames2Leg");
  phoTightLooseTrigNames_ = iConfig.getParameter<std::vector<std::string> >("phoTightLooseTrigNames");

  eleN1EffVars_ = iConfig.getParameter<std::vector<std::string> >("eleN1EffVars");
  eleSingleEffVars_ = iConfig.getParameter<std::vector<std::string> >("eleSingleEffVars");
  eleEffTags_ = iConfig.getParameter<std::vector<std::string> >("eleEffTags");
  eleTrigTPEffVsVars_ = iConfig.getParameter<std::vector<std::string> >("eleTrigTPEffVsVars");
  eleLooseTightTrigEffVsVars_ =  iConfig.getParameter<std::vector<std::string> >("eleLooseTightTrigEffVsVars");
  eleHLTvOfflineVars_ = iConfig.getParameter<std::vector<std::string> >("eleHLTvOfflineVars");

  phoN1EffVars_ = iConfig.getParameter<std::vector<std::string> >("phoN1EffVars");
  phoSingleEffVars_ = iConfig.getParameter<std::vector<std::string> >("phoSingleEffVars");
  phoEffTags_ = iConfig.getParameter<std::vector<std::string> >("phoEffTags");
  phoTrigTPEffVsVars_ = iConfig.getParameter<std::vector<std::string> >("phoTrigTPEffVsVars");
  phoLooseTightTrigEffVsVars_ =  iConfig.getParameter<std::vector<std::string> >("phoLooseTightTrigEffVsVars");
  phoHLTvOfflineVars_ = iConfig.getParameter<std::vector<std::string> >("phoHLTvOfflineVars");

  runClientEndLumiBlock_ = iConfig.getParameter<bool>("runClientEndLumiBlock");
  runClientEndRun_ = iConfig.getParameter<bool>("runClientEndRun");
  runClientEndJob_ = iConfig.getParameter<bool>("runClientEndJob");

  dirName_ = iConfig.getParameter<std::string>("DQMDirName");

  filterInactiveTriggers_ =iConfig.getParameter<bool>("filterInactiveTriggers");
  hltTag_ = iConfig.getParameter<std::string>("hltTag");


}


EgHLTOfflineClient::~EgHLTOfflineClient() { }


void EgHLTOfflineClient::dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) {

  if (runClientEndJob_) runClient_(ibooker_, igetter_);
}

void EgHLTOfflineClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  if (!isSetup_) {
    if (filterInactiveTriggers_) {
      HLTConfigProvider hltConfig;
      bool changed = false;
      hltConfig.init(run, c, hltTag_, changed);
      std::vector<std::string> activeFilters;
      std::vector<std::string> activeEleFilters;
      std::vector<std::string> activeEle2LegFilters;
      std::vector<std::string> activePhoFilters;
      std::vector<std::string> activePho2LegFilters;
      egHLT::trigTools::getActiveFilters(hltConfig, activeFilters, activeEleFilters,
          activeEle2LegFilters, activePhoFilters, activePho2LegFilters);

      egHLT::trigTools::filterInactiveTriggers(eleHLTFilterNames_, activeEleFilters);
      egHLT::trigTools::filterInactiveTriggers(eleHLTFilterNames2Leg_, activeEle2LegFilters);
      egHLT::trigTools::filterInactiveTriggers(phoHLTFilterNames_, activePhoFilters);
      egHLT::trigTools::filterInactiveTightLooseTriggers(eleTightLooseTrigNames_, activeEleFilters);
      egHLT::trigTools::filterInactiveTightLooseTriggers(phoTightLooseTrigNames_, activePhoFilters);
    }
    isSetup_ = true;
  }
}

void EgHLTOfflineClient::dqmEndLuminosityBlock( DQMStore::IBooker & ibooker_,
    DQMStore::IGetter & igetter_, edm::LuminosityBlock const & iLumi,
    edm::EventSetup const& iSetup) {

  if (runClientEndLumiBlock_)  runClient_(ibooker_, igetter_);
}

void EgHLTOfflineClient::runClient_(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter)
{

  ibooker.setCurrentFolder(dirName_ + "/Client_Histos");


  std::vector<std::string> regions;
  regions.push_back("eb");
  regions.push_back("ee");

  for (size_t filterNr = 0; filterNr < eleHLTFilterNames_.size(); filterNr++) {
    //std::cout<<"FilterName: "<<eleHLTFilterNames_[filterNr]<<std::endl;
    for (size_t regionNr = 0; regionNr < regions.size(); regionNr++) {
      for (size_t effNr = 0; effNr < eleEffTags_.size(); effNr++) {
        //----Morse----
        ibooker.setCurrentFolder(dirName_+"/Client_Histos/"+eleHLTFilterNames_[filterNr]);
        //--------------
	      createN1EffHists(eleHLTFilterNames_[filterNr], 
            eleHLTFilterNames_[filterNr] + "_gsfEle_" + eleEffTags_[effNr], regions[regionNr],
            eleN1EffVars_, ibooker, igetter);

        createSingleEffHists(eleHLTFilterNames_[filterNr],
            eleHLTFilterNames_[filterNr] + "_gsfEle_" + eleEffTags_[effNr], regions[regionNr],
            eleSingleEffVars_, ibooker, igetter);

        createTrigTagProbeEffHistsNewAlgo(eleHLTFilterNames_[filterNr], regions[regionNr],
            eleTrigTPEffVsVars_, "gsfEle", ibooker, igetter);

        createHLTvsOfflineHists(eleHLTFilterNames_[filterNr],
            eleHLTFilterNames_[filterNr] + "_gsfEle_passFilter", regions[regionNr],
            eleHLTvOfflineVars_, ibooker, igetter);

      }
    }
  }
  for (size_t filterNr = 0; filterNr < eleHLTFilterNames2Leg_.size(); filterNr++) {
    for (size_t regionNr = 0; regionNr < regions.size(); regionNr++) {
      for (size_t effNr = 0; effNr < eleEffTags_.size(); effNr++) {
        std::string trigNameLeg1 = eleHLTFilterNames2Leg_[filterNr].substr(
            0, eleHLTFilterNames2Leg_[filterNr].find("::"));

        std::string trigNameLeg2 = eleHLTFilterNames2Leg_[filterNr].substr(
            eleHLTFilterNames2Leg_[filterNr].find("::") + 2);

	      ibooker.setCurrentFolder(dirName_+"/Client_Histos/"+trigNameLeg2);
        createTrigTagProbeEffHists2Leg(trigNameLeg1, trigNameLeg2, regions[regionNr],
            eleTrigTPEffVsVars_, "gsfEle", ibooker, igetter);
      }
    }
  }

  for (size_t filterNr = 0; filterNr < phoHLTFilterNames_.size(); filterNr++) {
    for (size_t regionNr = 0; regionNr < regions.size(); regionNr++) {
      for (size_t effNr = 0; effNr < phoEffTags_.size(); effNr++) {
        //----Morse----
        ibooker.setCurrentFolder(dirName_+"/Client_Histos/"+phoHLTFilterNames_[filterNr]);
        createN1EffHists(phoHLTFilterNames_[filterNr], 
            phoHLTFilterNames_[filterNr] + "_pho_" + phoEffTags_[effNr], regions[regionNr],
            phoN1EffVars_, ibooker, igetter);

        createSingleEffHists(phoHLTFilterNames_[filterNr],
            phoHLTFilterNames_[filterNr] + "_pho_" + phoEffTags_[effNr], regions[regionNr],
            phoSingleEffVars_, ibooker, igetter);

        createTrigTagProbeEffHistsNewAlgo(phoHLTFilterNames_[filterNr], regions[regionNr],
            phoTrigTPEffVsVars_, "pho", ibooker, igetter);

        createHLTvsOfflineHists(phoHLTFilterNames_[filterNr],
            phoHLTFilterNames_[filterNr] + "_pho_passFilter", regions[regionNr],
            phoHLTvOfflineVars_, ibooker, igetter);

        //--------------
      }
    }
  }

  for (size_t regionNr = 0; regionNr < regions.size(); regionNr++) {
    createLooseTightTrigEff(eleTightLooseTrigNames_, regions[regionNr],
        eleLooseTightTrigEffVsVars_, "gsfEle", ibooker, igetter);

    createLooseTightTrigEff(eleTightLooseTrigNames_, regions[regionNr],
        eleLooseTightTrigEffVsVars_, "gsfEle_trigCuts", ibooker, igetter);

    createLooseTightTrigEff(phoTightLooseTrigNames_, regions[regionNr],
        phoLooseTightTrigEffVsVars_, "pho", ibooker, igetter);

    createLooseTightTrigEff(phoTightLooseTrigNames_, regions[regionNr],
        phoLooseTightTrigEffVsVars_, "pho_trigCuts", ibooker, igetter);

  }
  //----Morse-----
  ibooker.setCurrentFolder(dirName_);
  //----------
}

void EgHLTOfflineClient::createHLTvsOfflineHists(const std::string& filterName,
    const std::string& baseName, const std::string& region,
    const std::vector<std::string>& varNames, DQMStore::IBooker& ibooker,
    DQMStore::IGetter& igetter){

  //need to do Energy manually to get SC Energy
  /*
  MonitorElement* numer = dbe_->get(dirName_+"/Source_Histos/"+filterName+"/"+baseName+"_HLTenergy"+"_"+region);
  MonitorElement* denom = dbe_->get(dirName_+"/Source_Histos/"+filterName+"/"+baseName+"_energy"+"_"+region);

  if(numer!=NULL && denom!=NULL){
    std::string effHistName(baseName+"_HLToverOfflineSC_energy_"+region);//std::cout<<"hltVSoffline:  "<<effHistName<<std::endl;
    std::string effHistTitle(effHistName);
    if(region=="eb" || region=="ee"){
      if(region=="eb") effHistTitle = "Barrel "+baseName+" HLToverOfflineSC Energy";
      if(region=="ee") effHistTitle = "Endcap "+baseName+" HLToverOfflineSC Energy";
      FillHLTvsOfflineHist(filterName,effHistName,effHistTitle,numer,denom);	
    }
  }//end Et
  */

  //now eta, phi automatically
  for (size_t varNr = 0; varNr < varNames.size(); varNr++) {
    MonitorElement* numer = igetter.get(dirName_+"/Source_Histos/"+filterName+"/"+baseName+"_HLT"+varNames[varNr]+"_"+region);
    MonitorElement* denom = igetter.get(dirName_+"/Source_Histos/"+filterName+"/"+baseName+"_"+varNames[varNr]+"_"+region);
    if (numer != NULL && denom != NULL) {
      std::string effHistName(baseName + "_HLToverOffline_" + varNames[varNr] + "_" + region);//std::cout<<"hltVSoffline:  "<<effHistName<<std::endl;
      std::string effHistTitle(effHistName);
      if (region == "eb" || region == "ee") {
        if (region == "eb") effHistTitle = "Barrel " + baseName + " HLToverOffline " + varNames[varNr];
        if (region == "ee") effHistTitle = "Endcap " + baseName + " HLToverOffline " + varNames[varNr];
          FillHLTvsOfflineHist(filterName, effHistName, effHistTitle, numer, denom, ibooker, igetter);
      }
    }
  }//end loop over varNames 
}

MonitorElement* EgHLTOfflineClient::FillHLTvsOfflineHist(const std::string& filter,
    const std::string& name, const std::string& title, const MonitorElement* numer,
    const MonitorElement* denom, DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

  TH1F* num = numer->getTH1F();
  if (num->GetSumw2N() == 0) num->Sumw2();

  TH1F* den = denom->getTH1F();
  if (den->GetSumw2N() == 0) den->Sumw2();

  TH1F* h_eff = (TH1F*)num->Clone(name.c_str());
  h_eff->Divide(num, den, 1, 1, "B");
  h_eff->SetTitle(title.c_str());
  MonitorElement* eff = igetter.get(dirName_ + "/Client_Histos/" + filter + "/" + name);
  if (eff == NULL) {
    eff = ibooker.book1D(name, h_eff);
  } else { //I was having problems with collating the histograms, hence why I'm just resetting the histogram value
    *eff->getTH1F() = *h_eff;
    delete h_eff;
  }
  return eff;
}

void EgHLTOfflineClient::createN1EffHists(const std::string& filterName,
    const std::string& baseName, const std::string& region,
    const std::vector<std::string>& varNames, DQMStore::IBooker& ibooker,
    DQMStore::IGetter& igetter) {

  MonitorElement* numer = igetter.get(dirName_+"/Source_Histos/"+filterName+"/"+baseName+"_allCuts_"+region);

  for (size_t varNr = 0; varNr < varNames.size(); varNr++) {
    MonitorElement* denom = igetter.get(dirName_+"/Source_Histos/"+filterName+"/"+baseName+"_n1_"+varNames[varNr]+"_"+region);
    if (numer != NULL && denom != NULL) {
      std::string effHistName(baseName+"_n1Eff_"+varNames[varNr]+"_"+region);//std::cout<<"N1:  "<<effHistName<<std::endl;
      //std::cout<<region<<"  ";
      //----Morse-----------
      std::string effHistTitle(effHistName);//std::cout<<effHistTitle<<std::endl;
      if ( region == "eb" || region == "ee"){
        if (region == "eb") effHistTitle = "Barrel "+baseName+" N1eff "+varNames[varNr];
        if (region == "ee") effHistTitle = "Endcap "+baseName+" N1eff "+varNames[varNr];
      }//std::cout<<effHistTitle<<std::endl;
      makeEffMonElemFromPassAndAll(filterName, effHistName, effHistTitle, numer, denom, ibooker,
          igetter);
      //---------------------
    }
  }//end loop over varNames 
}

void EgHLTOfflineClient::createSingleEffHists(const std::string& filterName,
    const std::string& baseName, const std::string& region,
    const std::vector<std::string>& varNames, DQMStore::IBooker& ibooker,
    DQMStore::IGetter& igetter) {

  MonitorElement* denom = igetter.get(dirName_+"/Source_Histos/"+filterName+"/"+baseName+"_noCuts_"+region);

  for (size_t varNr = 0; varNr < varNames.size(); varNr++) {
    MonitorElement* numer = igetter.get(dirName_+"/Source_Histos/"+filterName+"/"+baseName+"_single_"+varNames[varNr]+"_"+region);
    if (numer != NULL && denom != NULL) {
      std::string effHistName(baseName + "_singleEff_" + varNames[varNr] + "_" + region);//std::cout<<"Si:  "<<effHistName<<std::endl;
      //----Morse-----------
      std::string effHistTitle(effHistName);//std::cout<<effHistTitle<<std::endl;
      if (region == "eb" || region == "ee") {
        if (region == "eb") effHistTitle = "Barrel " + baseName + " SingleEff " + varNames[varNr];
        if (region == "ee") effHistTitle = "Endcap " + baseName + " SingleEff " + varNames[varNr];
      }//std::cout<<effHistTitle<<std::endl;
      makeEffMonElemFromPassAndAll(filterName, effHistName, effHistTitle, numer, denom, ibooker,
          igetter);
      //--------------------
    }
  }//end loop over varNames 
}

void EgHLTOfflineClient::createTrigTagProbeEffHists(const std::string& filterName,
    const std::string& region, const std::vector<std::string>& vsVarNames,
    const std::string& objName, DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

  for (size_t varNr = 0; varNr < vsVarNames.size(); varNr++) {
    std::string allName(dirName_ + "/Source_Histos/" + filterName + "/" + filterName + "_trigTagProbe_" + objName + "_all_" + vsVarNames[varNr] + "_" + region);
    MonitorElement* all = igetter.get(allName);
    if (all == NULL) {
      continue;
    }
    std::string passName(dirName_ + "/Source_Histos/" + filterName + "/" + filterName + "_trigTagProbe_" + objName + "_pass_" + vsVarNames[varNr] + "_" + region);
    MonitorElement* pass = igetter.get(passName);
    if (pass == NULL) {
      continue;
    }
    //----Morse-----
    std::string effHistTitle(filterName + "_trigTagProbeEff_" + objName + "_vs_" + vsVarNames[varNr] + "_" + region);//std::cout<<effHistTitle<<std::endl;
    if (region == "eb" || region == "ee") {
      if (region == "eb") effHistTitle = "Barrel " + filterName + "_" + objName + " TrigTagProbeEff vs " + vsVarNames[varNr];
      if (region == "ee") effHistTitle = "Endcap " + filterName + "_" + objName + " TrigTagProbeEff vs " + vsVarNames[varNr];
    }
    //------------
    makeEffMonElemFromPassAndAll(filterName,
        filterName + "_trigTagProbeEff_" + objName + "_vs_" + vsVarNames[varNr] + "_" + region,
        effHistTitle, pass, all, ibooker, igetter);

  }//end loop over vsVarNames
}

void EgHLTOfflineClient::createTrigTagProbeEffHistsNewAlgo(const std::string& filterName,
    const std::string& region, const std::vector<std::string>& vsVarNames,
    const std::string& objName, DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

  for (size_t varNr = 0; varNr < vsVarNames.size(); varNr++) {
    /* 
       std::string allName(dirName_+"/Source_Histos/"+filterName+"/"+filterName+"_trigTagProbe_"+objName+"_all_"+vsVarNames[varNr]+"_"+region);
       MonitorElement* all = dbe_->get(allName); 
       if(all==NULL){
       //edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<allName;
       continue;
       }*/
    std::string passName(dirName_+"/Source_Histos/"+filterName+"/"+filterName+"_trigTagProbe_"+objName+"_passNotTag_"+vsVarNames[varNr]+"_"+region);
    MonitorElement* passNotTag = igetter.get(passName);
    if (passNotTag == NULL) {
      //edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<passName;
      continue;
    }
    std::string passTagTagName(dirName_+"/Source_Histos/"+filterName+"/"+filterName+"_trigTagProbe_"+objName+"_passTagTag_"+vsVarNames[varNr]+"_"+region);
    MonitorElement* passTagTag = igetter.get(passTagTagName);
    if (passTagTag == NULL) {
      //edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<passTagTagName;
      continue;
    }
    std::string failName(dirName_+"/Source_Histos/"+filterName+"/"+filterName+"_trigTagProbe_"+objName+"_fail_"+vsVarNames[varNr]+"_"+region);
    MonitorElement* fail = igetter.get(failName);
    if (fail == NULL) {
      //edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<failName;
      continue;
    }
    //----Morse-----
    std::string effHistTitle(filterName+"_trigTagProbeEff_"+objName+"_vs_"+vsVarNames[varNr]+"_"+region);//std::cout<<effHistTitle<<std::endl;
    if (region == "eb" || region == "ee") {
      if (region == "eb") effHistTitle = "Barrel " + filterName + "_"+objName + " TrigTagProbeEff vs " + vsVarNames[varNr];
      if (region == "ee") effHistTitle = "Endcap " + filterName + "_"+objName + " TrigTagProbeEff vs " + vsVarNames[varNr];
    }//std::cout<<effHistTitle<<std::endl;
    //------------
    makeEffMonElemFromPassAndFailAndTagTag(filterName,
        filterName + "_trigTagProbeEff_" + objName + "_vs_" + vsVarNames[varNr] + "_" + region,
        effHistTitle, passNotTag, fail, passTagTag, ibooker, igetter);
  }//end loop over vsVarNames
}

void EgHLTOfflineClient::createTrigTagProbeEffHists2Leg(const std::string& filterNameLeg1,
    const std::string& filterNameLeg2, const std::string& region,
    const std::vector<std::string>& vsVarNames, const std::string& objName,
    DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

  for (size_t varNr = 0; varNr < vsVarNames.size(); varNr++) {

    std::string allName(dirName_+"/Source_Histos/"+filterNameLeg2+"/"+filterNameLeg2+"_trigTagProbe_"+objName+"_all_"+vsVarNames[varNr]+"_"+region);
    MonitorElement* all = igetter.get(allName);
    if (all == NULL) {
      edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<allName;
      continue;
    }

    std::string Leg2NotLeg1SourceName(dirName_+"/Source_Histos/"+filterNameLeg2+"/"+filterNameLeg2+"_trigTagProbe_"+objName+"_passLeg2failLeg1_"+vsVarNames[varNr]+"_"+region);
    MonitorElement* Leg2NotLeg1Source = igetter.get(Leg2NotLeg1SourceName);
    if (Leg2NotLeg1Source == NULL) {
      edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<Leg2NotLeg1SourceName;
      continue;
    }

    std::string Leg1EffName(dirName_+"/Client_Histos/"+filterNameLeg1+"/"+filterNameLeg1+"_trigTagProbeEff_"+objName+"_vs_"+vsVarNames[varNr]+"_"+region);
    MonitorElement *Leg1Eff = igetter.get(Leg1EffName);
    if (Leg1Eff == NULL) {
      edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<Leg1EffName;
      continue;
    }

    std::string effHistTitle(filterNameLeg2+"_trigTagProbeEff2Leg_"+objName+"_vs_"+vsVarNames[varNr]+"_"+region);//std::cout<<effHistTitle<<std::endl;
    if (region == "eb" || region == "ee") {
      if (region == "eb") effHistTitle = "Barrel " + filterNameLeg2 + "_" + objName + " TrigTagProbeEff2Leg vs " + vsVarNames[varNr];
      if (region == "ee") effHistTitle = "Endcap " + filterNameLeg2 + "_" + objName + " TrigTagProbeEff2Leg vs " + vsVarNames[varNr];
    }//std::cout<<effHistTitle<<std::endl;
    makeEffMonElem2Leg(filterNameLeg2,
        filterNameLeg2 + "_trigTagProbeEff2Leg_" + objName + "_vs_" + vsVarNames[varNr] + "_" + region,
        effHistTitle, Leg1Eff, Leg2NotLeg1Source, all, ibooker, igetter);
  }//end loop over vsVarNames
}


void EgHLTOfflineClient::createLooseTightTrigEff(
    const std::vector<std::string>&  tightLooseTrigNames, const std::string& region,
    const std::vector<std::string>& vsVarNames, const std::string& objName,
    DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

  for (size_t varNr = 0; varNr < vsVarNames.size(); varNr++) {
    for (size_t trigNr = 0; trigNr < tightLooseTrigNames.size(); trigNr++) {
      std::vector<std::string> splitString;
      boost::split(splitString, tightLooseTrigNames[trigNr], boost::is_any_of(std::string(":")));
      if (splitString.size() != 2) {
        continue; //format incorrect 
      }

      const std::string& tightTrig = splitString[0];
      const std::string& looseTrig = splitString[1];
      MonitorElement* fail = igetter.get(dirName_ + "/Source_Histos/" + tightTrig + "_" + looseTrig + "_" + objName + "_failTrig_" + vsVarNames[varNr] + "_" + region);
      if (fail == NULL) {
        continue;
      }

      MonitorElement* pass = igetter.get(dirName_ + "/Source_Histos/" + tightTrig + "_" + looseTrig + "_" + objName + "_passTrig_" + vsVarNames[varNr] + "_" + region);
      if (pass == NULL) {
        continue;
      }

      const std::string newHistName(tightTrig + "_trigEffTo_" + looseTrig + "_" + objName + "_vs_" + vsVarNames[varNr] + "_" + region);
      //----Morse-----
      std::string effHistTitle(newHistName);//std::cout<<effHistTitle<<std::endl;
      if (region == "eb" || region == "ee") {
        if (region == "eb") effHistTitle = "Barrel " + tightTrig + "_TrigEffTo_" + looseTrig + "_" + objName + " vs " + vsVarNames[varNr];
        if (region == "ee") effHistTitle = "Endcap " + tightTrig + "_TrigEffTo_" + looseTrig + "_" + objName + " vs " + vsVarNames[varNr];
      }
      //------------
      makeEffMonElemFromPassAndFail("LooseTight", newHistName, effHistTitle, pass, fail,
          ibooker, igetter);

    }//end loop over trigger pairs
  } //end loop over vsVarNames
  
}
//-----Morse-------
MonitorElement* EgHLTOfflineClient::makeEffMonElemFromPassAndAll(const std::string& filterName,
    const std::string& name, const std::string& title, const MonitorElement* pass,
    const MonitorElement* all, DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

  TH1F* passHist = pass->getTH1F();
  if (passHist->GetSumw2N() == 0) passHist->Sumw2();
  TH1F* allHist = all->getTH1F();
  if (allHist->GetSumw2N() == 0) allHist->Sumw2();
  TH1F* effHist = (TH1F*) passHist->Clone(name.c_str());
  effHist->Divide(passHist,allHist,1,1,"B");
  //----Morse---------
  effHist->SetTitle(title.c_str());
  //------------------
  MonitorElement* eff = igetter.get(dirName_+"/Client_Histos/"+filterName+"/"+name);
  if (eff == NULL) {
    eff = ibooker.book1D(name,effHist);
  } else { //I was having problems with collating the histograms, hence why I'm just resetting the histogram value
    *eff->getTH1F()=*effHist;
    delete effHist;
  }
  return eff;
}

MonitorElement* EgHLTOfflineClient::makeEffMonElemFromPassAndFailAndTagTag(
    const std::string& filter, const std::string& name, const std::string& title,
    const MonitorElement* passNotTag, const MonitorElement* fail, const MonitorElement* tagtag,
    DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

  TH1F* passHist = passNotTag->getTH1F();
  if (passHist->GetSumw2N() == 0) passHist->Sumw2();
  TH1F* failHist = fail->getTH1F();
  if (failHist->GetSumw2N() == 0) failHist->Sumw2();
  TH1F* tagtagHist = tagtag->getTH1F();
  if (tagtagHist->GetSumw2N() == 0) tagtagHist->Sumw2();
  TH1F* numer = (TH1F*) passHist->Clone(name.c_str());
  if (numer->GetSumw2N() == 0) numer->Sumw2();
  numer->Add(tagtagHist,passHist,2,1);
  TH1F* denom = (TH1F*) passHist->Clone(name.c_str());
  if (denom->GetSumw2N() == 0) denom->Sumw2();
  denom->Add(tagtagHist,passHist,2,1);
  denom->Add(failHist,1);
  TH1F* effHist = (TH1F*) passHist->Clone(name.c_str());
  //TGraphAsymmErrors *effHist = new TGraphAsymmErrors(numer,denom,"cl=0.683 b(1,1) mode");
  effHist->Divide(numer, denom, 1, 1, "B");
  //effHist->Divide(numer,denom,"cl=0.683 b(1,1) mode");
  //----Morse---------
  effHist->SetTitle(title.c_str());
  //------------------
  MonitorElement* eff = igetter.get(dirName_+"/Client_Histos/"+filter+"/"+name);
  if (eff == NULL) {
    eff = ibooker.book1D(name, effHist);
  }
  else { //I was having problems with collating the histograms, hence why I'm just resetting the histogram value
    *eff->getTH1F()=*effHist;
    //*eff->getTGraphAsymmErrors()=*effHist;
    delete effHist;
  }
  return eff;
}

MonitorElement* EgHLTOfflineClient::makeEffMonElem2Leg(const std::string& filter,
    const std::string& name, const std::string& title, const MonitorElement* Leg1Eff,
    const MonitorElement* Leg2NotLeg1Source, const MonitorElement* all, DQMStore::IBooker& ibooker,
    DQMStore::IGetter& igetter) {

  TH1F* allHist = all->getTH1F();
  if (allHist->GetSumw2() == 0) allHist->Sumw2();
  TH1F* Leg2NotLeg1SourceHist = Leg2NotLeg1Source->getTH1F();
  if (Leg2NotLeg1SourceHist->GetSumw2() == 0) Leg2NotLeg1SourceHist->Sumw2();

  TH1F* effHistLeg2NotLeg1 = (TH1F*)allHist->Clone(name.c_str());
  if (effHistLeg2NotLeg1->GetSumw2() == 0) effHistLeg2NotLeg1->Sumw2();
  effHistLeg2NotLeg1->Divide(Leg2NotLeg1SourceHist, allHist, 1, 1, "B");

  TH1F* Leg1EffHist = Leg1Eff->getTH1F();
  if (Leg1EffHist->GetSumw2() == 0) Leg1EffHist->Sumw2();

  TH1F* effHistTerm1 = (TH1F*)allHist->Clone(name.c_str());
  if (effHistTerm1->GetSumw2() == 0) effHistTerm1->Sumw2();
  effHistTerm1->Multiply(Leg1EffHist, Leg1EffHist, 1, 1, "B");

  TH1F* effHistTerm2 = (TH1F*)allHist->Clone(name.c_str());
  if (effHistTerm2->GetSumw2() == 0) effHistTerm2->Sumw2();
  effHistTerm2->Multiply(Leg1EffHist, effHistLeg2NotLeg1, 1, 1, "B");
  effHistTerm2->Scale(2);

  TH1F* effHist = (TH1F*)allHist->Clone(name.c_str());
  if (effHist->GetSumw2() == 0) effHist->Sumw2();
  effHist->Add(effHistTerm1, effHistTerm2, 1, 1);
  effHist->SetTitle(title.c_str());
  
  MonitorElement* eff = igetter.get(dirName_ + "/Client_Histos/" + filter + "/" + name);
  if (eff == NULL) {
    eff = ibooker.book1D(name, effHist); 
  } else { //I was having problems with collating the histograms, hence why I'm just resetting the histogram value
    *eff->getTH1F() = *effHist; 
    delete effHist;
  }
  return eff;
}

//-----Morse-------
MonitorElement* EgHLTOfflineClient::makeEffMonElemFromPassAndFail(const std::string& filterName,
    const std::string& name, const std::string& title, const MonitorElement* pass,
    const MonitorElement* fail, DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

  TH1F* failHist = fail->getTH1F();
  if (failHist->GetSumw2N() == 0) failHist->Sumw2();
  TH1F* passHist = pass->getTH1F();
  if (passHist->GetSumw2N() == 0) passHist->Sumw2();

  TH1F* effHist = (TH1F*) passHist->Clone(name.c_str());
  effHist->Add(failHist);
  effHist->Divide(passHist, effHist, 1, 1, "B");
  //----Morse---------
  effHist->SetTitle(title.c_str());
  //------------------  
  MonitorElement* eff = igetter.get(dirName_ + "/Client_Histos/" + filterName + "/" + name);
  if (eff == NULL) {
    eff = ibooker.book1D(name, effHist);
  } else { //I was having problems with collating the histograms, hence why I'm just reseting the histogram value
    *eff->getTH1F() = *effHist;
    delete effHist;
  }
  return eff;
}
