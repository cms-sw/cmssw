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


EgHLTOfflineClient::~EgHLTOfflineClient() = default;


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
  regions.emplace_back("eb");
  regions.emplace_back("ee");

  for (auto const & eleHLTFilterName : eleHLTFilterNames_) {
    //std::cout<<"FilterName: "<<eleHLTFilterNames_[filterNr]<<std::endl;
    for (auto const & region : regions) {
      for (auto const & eleEffTag : eleEffTags_) {
        //----Morse----
        ibooker.setCurrentFolder(dirName_+"/Client_Histos/"+eleHLTFilterName);
        //--------------
	      createN1EffHists(eleHLTFilterName, 
            eleHLTFilterName + "_gsfEle_" + eleEffTag, region,
            eleN1EffVars_, ibooker, igetter);

        createSingleEffHists(eleHLTFilterName,
            eleHLTFilterName + "_gsfEle_" + eleEffTag, region,
            eleSingleEffVars_, ibooker, igetter);

        createTrigTagProbeEffHistsNewAlgo(eleHLTFilterName, region,
            eleTrigTPEffVsVars_, "gsfEle", ibooker, igetter);

        createHLTvsOfflineHists(eleHLTFilterName,
            eleHLTFilterName + "_gsfEle_passFilter", region,
            eleHLTvOfflineVars_, ibooker, igetter);

      }
    }
  }
  for (auto & filterNr : eleHLTFilterNames2Leg_) {
    for (auto const & region : regions) {
      for (size_t effNr = 0; effNr < eleEffTags_.size(); effNr++) {
        std::string trigNameLeg1 = filterNr.substr(
            0, filterNr.find("::"));

        std::string trigNameLeg2 = filterNr.substr(
            filterNr.find("::") + 2);

	      ibooker.setCurrentFolder(dirName_+"/Client_Histos/"+trigNameLeg2);
        createTrigTagProbeEffHists2Leg(trigNameLeg1, trigNameLeg2, region,
            eleTrigTPEffVsVars_, "gsfEle", ibooker, igetter);
      }
    }
  }

  for (auto const & phoHLTFilterName : phoHLTFilterNames_) {
    for (auto const & region : regions) {
      for (auto const & phoEffTag : phoEffTags_) {
        //----Morse----
        ibooker.setCurrentFolder(dirName_+"/Client_Histos/"+phoHLTFilterName);
        createN1EffHists(phoHLTFilterName, 
            phoHLTFilterName + "_pho_" + phoEffTag, region,
            phoN1EffVars_, ibooker, igetter);

        createSingleEffHists(phoHLTFilterName,
            phoHLTFilterName + "_pho_" + phoEffTag, region,
            phoSingleEffVars_, ibooker, igetter);

        createTrigTagProbeEffHistsNewAlgo(phoHLTFilterName, region,
            phoTrigTPEffVsVars_, "pho", ibooker, igetter);

        createHLTvsOfflineHists(phoHLTFilterName,
            phoHLTFilterName + "_pho_passFilter", region,
            phoHLTvOfflineVars_, ibooker, igetter);

        //--------------
      }
    }
  }

  for (auto const & region : regions) {
    createLooseTightTrigEff(eleTightLooseTrigNames_, region,
        eleLooseTightTrigEffVsVars_, "gsfEle", ibooker, igetter);

    createLooseTightTrigEff(eleTightLooseTrigNames_, region,
        eleLooseTightTrigEffVsVars_, "gsfEle_trigCuts", ibooker, igetter);

    createLooseTightTrigEff(phoTightLooseTrigNames_, region,
        phoLooseTightTrigEffVsVars_, "pho", ibooker, igetter);

    createLooseTightTrigEff(phoTightLooseTrigNames_, region,
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
  for (auto const & varName : varNames) {
    MonitorElement* numer = igetter.get(dirName_+"/Source_Histos/"+filterName+"/"+baseName+"_HLT"+varName+"_"+region);
    MonitorElement* denom = igetter.get(dirName_+"/Source_Histos/"+filterName+"/"+baseName+"_"+varName+"_"+region);
    if (numer != nullptr && denom != nullptr) {
      std::string effHistName(baseName + "_HLToverOffline_" + varName + "_" + region);//std::cout<<"hltVSoffline:  "<<effHistName<<std::endl;
      std::string effHistTitle(effHistName);
      if (region == "eb" || region == "ee") {
        if (region == "eb") effHistTitle = "Barrel " + baseName + " HLToverOffline " + varName;
        if (region == "ee") effHistTitle = "Endcap " + baseName + " HLToverOffline " + varName;
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

  auto* h_eff = (TH1F*)num->Clone(name.c_str());
  h_eff->Divide(num, den, 1, 1, "B");
  h_eff->SetTitle(title.c_str());
  MonitorElement* eff = igetter.get(dirName_ + "/Client_Histos/" + filter + "/" + name);
  if (eff == nullptr) {
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

  for (auto const & varName : varNames) {
    MonitorElement* denom = igetter.get(dirName_+"/Source_Histos/"+filterName+"/"+baseName+"_n1_"+varName+"_"+region);
    if (numer != nullptr && denom != nullptr) {
      std::string effHistName(baseName+"_n1Eff_"+varName+"_"+region);//std::cout<<"N1:  "<<effHistName<<std::endl;
      //std::cout<<region<<"  ";
      //----Morse-----------
      std::string effHistTitle(effHistName);//std::cout<<effHistTitle<<std::endl;
      if ( region == "eb" || region == "ee"){
        if (region == "eb") effHistTitle = "Barrel "+baseName+" N1eff "+varName;
        if (region == "ee") effHistTitle = "Endcap "+baseName+" N1eff "+varName;
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

  for (auto const & varName : varNames) {
    MonitorElement* numer = igetter.get(dirName_+"/Source_Histos/"+filterName+"/"+baseName+"_single_"+varName+"_"+region);
    if (numer != nullptr && denom != nullptr) {
      std::string effHistName(baseName + "_singleEff_" + varName + "_" + region);//std::cout<<"Si:  "<<effHistName<<std::endl;
      //----Morse-----------
      std::string effHistTitle(effHistName);//std::cout<<effHistTitle<<std::endl;
      if (region == "eb" || region == "ee") {
        if (region == "eb") effHistTitle = "Barrel " + baseName + " SingleEff " + varName;
        if (region == "ee") effHistTitle = "Endcap " + baseName + " SingleEff " + varName;
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

  for (auto const & vsVarName : vsVarNames) {
    std::string allName(dirName_ + "/Source_Histos/" + filterName + "/" + filterName + "_trigTagProbe_" + objName + "_all_" + vsVarName + "_" + region);
    MonitorElement* all = igetter.get(allName);
    if (all == nullptr) {
      continue;
    }
    std::string passName(dirName_ + "/Source_Histos/" + filterName + "/" + filterName + "_trigTagProbe_" + objName + "_pass_" + vsVarName + "_" + region);
    MonitorElement* pass = igetter.get(passName);
    if (pass == nullptr) {
      continue;
    }
    //----Morse-----
    std::string effHistTitle(filterName + "_trigTagProbeEff_" + objName + "_vs_" + vsVarName + "_" + region);//std::cout<<effHistTitle<<std::endl;
    if (region == "eb" || region == "ee") {
      if (region == "eb") effHistTitle = "Barrel " + filterName + "_" + objName + " TrigTagProbeEff vs " + vsVarName;
      if (region == "ee") effHistTitle = "Endcap " + filterName + "_" + objName + " TrigTagProbeEff vs " + vsVarName;
    }
    //------------
    makeEffMonElemFromPassAndAll(filterName,
        filterName + "_trigTagProbeEff_" + objName + "_vs_" + vsVarName + "_" + region,
        effHistTitle, pass, all, ibooker, igetter);

  }//end loop over vsVarNames
}

void EgHLTOfflineClient::createTrigTagProbeEffHistsNewAlgo(const std::string& filterName,
    const std::string& region, const std::vector<std::string>& vsVarNames,
    const std::string& objName, DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

  for (auto const & vsVarName : vsVarNames) {
    /* 
       std::string allName(dirName_+"/Source_Histos/"+filterName+"/"+filterName+"_trigTagProbe_"+objName+"_all_"+vsVarNames[varNr]+"_"+region);
       MonitorElement* all = dbe_->get(allName); 
       if(all==NULL){
       //edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<allName;
       continue;
       }*/
    std::string passName(dirName_+"/Source_Histos/"+filterName+"/"+filterName+"_trigTagProbe_"+objName+"_passNotTag_"+vsVarName+"_"+region);
    MonitorElement* passNotTag = igetter.get(passName);
    if (passNotTag == nullptr) {
      //edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<passName;
      continue;
    }
    std::string passTagTagName(dirName_+"/Source_Histos/"+filterName+"/"+filterName+"_trigTagProbe_"+objName+"_passTagTag_"+vsVarName+"_"+region);
    MonitorElement* passTagTag = igetter.get(passTagTagName);
    if (passTagTag == nullptr) {
      //edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<passTagTagName;
      continue;
    }
    std::string failName(dirName_+"/Source_Histos/"+filterName+"/"+filterName+"_trigTagProbe_"+objName+"_fail_"+vsVarName+"_"+region);
    MonitorElement* fail = igetter.get(failName);
    if (fail == nullptr) {
      //edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<failName;
      continue;
    }
    //----Morse-----
    std::string effHistTitle(filterName+"_trigTagProbeEff_"+objName+"_vs_"+vsVarName+"_"+region);//std::cout<<effHistTitle<<std::endl;
    if (region == "eb" || region == "ee") {
      if (region == "eb") effHistTitle = "Barrel " + filterName + "_"+objName + " TrigTagProbeEff vs " + vsVarName;
      if (region == "ee") effHistTitle = "Endcap " + filterName + "_"+objName + " TrigTagProbeEff vs " + vsVarName;
    }//std::cout<<effHistTitle<<std::endl;
    //------------
    makeEffMonElemFromPassAndFailAndTagTag(filterName,
        filterName + "_trigTagProbeEff_" + objName + "_vs_" + vsVarName + "_" + region,
        effHistTitle, passNotTag, fail, passTagTag, ibooker, igetter);
  }//end loop over vsVarNames
}

void EgHLTOfflineClient::createTrigTagProbeEffHists2Leg(const std::string& filterNameLeg1,
    const std::string& filterNameLeg2, const std::string& region,
    const std::vector<std::string>& vsVarNames, const std::string& objName,
    DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

  for (auto const & vsVarName : vsVarNames) {

    std::string allName(dirName_+"/Source_Histos/"+filterNameLeg2+"/"+filterNameLeg2+"_trigTagProbe_"+objName+"_all_"+vsVarName+"_"+region);
    MonitorElement* all = igetter.get(allName);
    if (all == nullptr) {
      edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<allName;
      continue;
    }

    std::string Leg2NotLeg1SourceName(dirName_+"/Source_Histos/"+filterNameLeg2+"/"+filterNameLeg2+"_trigTagProbe_"+objName+"_passLeg2failLeg1_"+vsVarName+"_"+region);
    MonitorElement* Leg2NotLeg1Source = igetter.get(Leg2NotLeg1SourceName);
    if (Leg2NotLeg1Source == nullptr) {
      edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<Leg2NotLeg1SourceName;
      continue;
    }

    std::string Leg1EffName(dirName_+"/Client_Histos/"+filterNameLeg1+"/"+filterNameLeg1+"_trigTagProbeEff_"+objName+"_vs_"+vsVarName+"_"+region);
    MonitorElement *Leg1Eff = igetter.get(Leg1EffName);
    if (Leg1Eff == nullptr) {
      edm::LogInfo("EgHLTOfflineClient") <<" couldnt get hist "<<Leg1EffName;
      continue;
    }

    std::string effHistTitle(filterNameLeg2+"_trigTagProbeEff2Leg_"+objName+"_vs_"+vsVarName+"_"+region);//std::cout<<effHistTitle<<std::endl;
    if (region == "eb" || region == "ee") {
      if (region == "eb") effHistTitle = "Barrel " + filterNameLeg2 + "_" + objName + " TrigTagProbeEff2Leg vs " + vsVarName;
      if (region == "ee") effHistTitle = "Endcap " + filterNameLeg2 + "_" + objName + " TrigTagProbeEff2Leg vs " + vsVarName;
    }//std::cout<<effHistTitle<<std::endl;
    makeEffMonElem2Leg(filterNameLeg2,
        filterNameLeg2 + "_trigTagProbeEff2Leg_" + objName + "_vs_" + vsVarName + "_" + region,
        effHistTitle, Leg1Eff, Leg2NotLeg1Source, all, ibooker, igetter);
  }//end loop over vsVarNames
}


void EgHLTOfflineClient::createLooseTightTrigEff(
    const std::vector<std::string>&  tightLooseTrigNames, const std::string& region,
    const std::vector<std::string>& vsVarNames, const std::string& objName,
    DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

  for (auto const & vsVarName : vsVarNames) {
    for (auto const & tightLooseTrigName : tightLooseTrigNames) {
      std::vector<std::string> splitString;
      boost::split(splitString, tightLooseTrigName, boost::is_any_of(std::string(":")));
      if (splitString.size() != 2) {
        continue; //format incorrect 
      }

      const std::string& tightTrig = splitString[0];
      const std::string& looseTrig = splitString[1];
      MonitorElement* fail = igetter.get(dirName_ + "/Source_Histos/" + tightTrig + "_" + looseTrig + "_" + objName + "_failTrig_" + vsVarName + "_" + region);
      if (fail == nullptr) {
        continue;
      }

      MonitorElement* pass = igetter.get(dirName_ + "/Source_Histos/" + tightTrig + "_" + looseTrig + "_" + objName + "_passTrig_" + vsVarName + "_" + region);
      if (pass == nullptr) {
        continue;
      }

      const std::string newHistName(tightTrig + "_trigEffTo_" + looseTrig + "_" + objName + "_vs_" + vsVarName + "_" + region);
      //----Morse-----
      std::string effHistTitle(newHistName);//std::cout<<effHistTitle<<std::endl;
      if (region == "eb" || region == "ee") {
        if (region == "eb") effHistTitle = "Barrel " + tightTrig + "_TrigEffTo_" + looseTrig + "_" + objName + " vs " + vsVarName;
        if (region == "ee") effHistTitle = "Endcap " + tightTrig + "_TrigEffTo_" + looseTrig + "_" + objName + " vs " + vsVarName;
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
  auto* effHist = (TH1F*) passHist->Clone(name.c_str());
  effHist->Divide(passHist,allHist,1,1,"B");
  //----Morse---------
  effHist->SetTitle(title.c_str());
  //------------------
  MonitorElement* eff = igetter.get(dirName_+"/Client_Histos/"+filterName+"/"+name);
  if (eff == nullptr) {
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
  auto* numer = (TH1F*) passHist->Clone(name.c_str());
  if (numer->GetSumw2N() == 0) numer->Sumw2();
  numer->Add(tagtagHist,passHist,2,1);
  auto* denom = (TH1F*) passHist->Clone(name.c_str());
  if (denom->GetSumw2N() == 0) denom->Sumw2();
  denom->Add(tagtagHist,passHist,2,1);
  denom->Add(failHist,1);
  auto* effHist = (TH1F*) passHist->Clone(name.c_str());
  //TGraphAsymmErrors *effHist = new TGraphAsymmErrors(numer,denom,"cl=0.683 b(1,1) mode");
  effHist->Divide(numer, denom, 1, 1, "B");
  //effHist->Divide(numer,denom,"cl=0.683 b(1,1) mode");
  //----Morse---------
  effHist->SetTitle(title.c_str());
  //------------------
  MonitorElement* eff = igetter.get(dirName_+"/Client_Histos/"+filter+"/"+name);
  if (eff == nullptr) {
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
  if (allHist->GetSumw2() == nullptr) allHist->Sumw2();
  TH1F* Leg2NotLeg1SourceHist = Leg2NotLeg1Source->getTH1F();
  if (Leg2NotLeg1SourceHist->GetSumw2() == nullptr) Leg2NotLeg1SourceHist->Sumw2();

  auto* effHistLeg2NotLeg1 = (TH1F*)allHist->Clone(name.c_str());
  if (effHistLeg2NotLeg1->GetSumw2() == nullptr) effHistLeg2NotLeg1->Sumw2();
  effHistLeg2NotLeg1->Divide(Leg2NotLeg1SourceHist, allHist, 1, 1, "B");

  TH1F* Leg1EffHist = Leg1Eff->getTH1F();
  if (Leg1EffHist->GetSumw2() == nullptr) Leg1EffHist->Sumw2();

  auto* effHistTerm1 = (TH1F*)allHist->Clone(name.c_str());
  if (effHistTerm1->GetSumw2() == nullptr) effHistTerm1->Sumw2();
  effHistTerm1->Multiply(Leg1EffHist, Leg1EffHist, 1, 1, "B");

  auto* effHistTerm2 = (TH1F*)allHist->Clone(name.c_str());
  if (effHistTerm2->GetSumw2() == nullptr) effHistTerm2->Sumw2();
  effHistTerm2->Multiply(Leg1EffHist, effHistLeg2NotLeg1, 1, 1, "B");
  effHistTerm2->Scale(2);

  auto* effHist = (TH1F*)allHist->Clone(name.c_str());
  if (effHist->GetSumw2() == nullptr) effHist->Sumw2();
  effHist->Add(effHistTerm1, effHistTerm2, 1, 1);
  effHist->SetTitle(title.c_str());
  
  MonitorElement* eff = igetter.get(dirName_ + "/Client_Histos/" + filter + "/" + name);
  if (eff == nullptr) {
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

  auto* effHist = (TH1F*) passHist->Clone(name.c_str());
  effHist->Add(failHist);
  effHist->Divide(passHist, effHist, 1, 1, "B");
  //----Morse---------
  effHist->SetTitle(title.c_str());
  //------------------  
  MonitorElement* eff = igetter.get(dirName_ + "/Client_Histos/" + filterName + "/" + name);
  if (eff == nullptr) {
    eff = ibooker.book1D(name, effHist);
  } else { //I was having problems with collating the histograms, hence why I'm just reseting the histogram value
    *eff->getTH1F() = *effHist;
    delete effHist;
  }
  return eff;
}
