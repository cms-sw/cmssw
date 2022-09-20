#include "DQMOffline/Trigger/interface/EgHLTOfflineSource.h"

#include "DQMOffline/Trigger/interface/EgHLTEleHLTFilterMon.h"
#include "DQMOffline/Trigger/interface/EgHLTPhoHLTFilterMon.h"

#include "DQMOffline/Trigger/interface/EgHLTDebugFuncs.h"
#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"
#include "DQMOffline/Trigger/interface/EgHLTTrigTools.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Run.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <boost/algorithm/string.hpp>

//#include "DQMOffline/Trigger/interface/EgHLTCutCodes.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
using namespace egHLT;

EgHLTOfflineSource::EgHLTOfflineSource(const edm::ParameterSet& iConfig) : nrEventsProcessed_(0) {
  binData_.setup(iConfig.getParameter<edm::ParameterSet>("binData"));
  cutMasks_.setup(iConfig.getParameter<edm::ParameterSet>("cutMasks"));
  eleHLTFilterNames_ = iConfig.getParameter<std::vector<std::string> >("eleHLTFilterNames");
  eleHLTFilterNames2Leg_ = iConfig.getParameter<std::vector<std::string> >("eleHLTFilterNames2Leg");
  phoHLTFilterNames_ = iConfig.getParameter<std::vector<std::string> >("phoHLTFilterNames");
  eleTightLooseTrigNames_ = iConfig.getParameter<std::vector<std::string> >("eleTightLooseTrigNames");
  diEleTightLooseTrigNames_ = iConfig.getParameter<std::vector<std::string> >("diEleTightLooseTrigNames");
  phoTightLooseTrigNames_ = iConfig.getParameter<std::vector<std::string> >("phoTightLooseTrigNames");
  diPhoTightLooseTrigNames_ = iConfig.getParameter<std::vector<std::string> >("diPhoTightLooseTrigNames");

  filterInactiveTriggers_ = iConfig.getParameter<bool>("filterInactiveTriggers");
  hltTag_ = iConfig.getParameter<std::string>("hltTag");
  dohep_ = iConfig.getParameter<bool>("doHEP");

  dirName_ = iConfig.getParameter<std::string>(
      "DQMDirName");  //"HLT/EgHLTOfflineSource_" + iConfig.getParameter<std::string>("@module_label");

  subdirName_ = iConfig.getParameter<std::string>("subDQMDirName");

  offEvtHelper_.setup(iConfig, consumesCollector());
}

EgHLTOfflineSource::~EgHLTOfflineSource() {
  // LogDebug("EgHLTOfflineSource") << "destructor called";
  for (auto& eleFilterMonHist : eleFilterMonHists_) {
    delete eleFilterMonHist;
  }
  for (auto& phoFilterMonHist : phoFilterMonHists_) {
    delete phoFilterMonHist;
  }
  for (auto& eleMonElem : eleMonElems_) {
    delete eleMonElem;
  }
  for (auto& phoMonElem : phoMonElems_) {
    delete phoMonElem;
  }
}

void EgHLTOfflineSource::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& c) {
  iBooker.setCurrentFolder(dirName_);

  //the one monitor element the source fills directly
  dqmErrsMonElem_ = iBooker.book1D("dqmErrors", "EgHLTOfflineSource Errors", 101, -0.5, 100.5);
  nrEventsProcessedMonElem_ = iBooker.bookInt("nrEventsProcessed");

  //if the HLTConfig changes during the job, the results are "un predictable" but in practice should be fine
  //the HLTConfig is used for working out which triggers are active, working out which filternames correspond to paths and L1 seeds
  //assuming those dont change for E/g it *should* be fine
  HLTConfigProvider hltConfig;
  bool changed = false;
  hltConfig.init(run, c, hltTag_, changed);
  if (filterInactiveTriggers_)
    filterTriggers(hltConfig);

  std::vector<std::string> hltFiltersUsed;
  getHLTFilterNamesUsed(hltFiltersUsed);
  trigCodes.reset(TrigCodes::makeCodes(hltFiltersUsed));

  offEvtHelper_.setupTriggers(hltConfig, hltFiltersUsed, *trigCodes);

  MonElemFuncs monElemFuncs(iBooker, *trigCodes);

  //now book ME's
  iBooker.setCurrentFolder(dirName_ + "/" + subdirName_);
  //each trigger path with generate object distributions and efficiencies (BUT not trigger efficiencies...)
  for (auto const& eleHLTFilterName : eleHLTFilterNames_) {
    iBooker.setCurrentFolder(dirName_ + "/" + subdirName_ + "/" + eleHLTFilterName);
    addEleTrigPath(monElemFuncs, eleHLTFilterName);
  }
  for (auto const& phoHLTFilterName : phoHLTFilterNames_) {
    iBooker.setCurrentFolder(dirName_ + "/" + subdirName_ + "/" + phoHLTFilterName);
    addPhoTrigPath(monElemFuncs, phoHLTFilterName);
  }
  //efficiencies of one trigger path relative to another
  monElemFuncs.initTightLooseTrigHists(eleMonElems_, eleTightLooseTrigNames_, binData_, "gsfEle");
  //new EgHLTDQMVarCut<OffEle>(cutMasks_.stdEle,&OffEle::cutCode));
  //monElemFuncs.initTightLooseTrigHistsTrigCuts(eleMonElems_,eleTightLooseTrigNames_,binData_);

  monElemFuncs.initTightLooseTrigHists(phoMonElems_, phoTightLooseTrigNames_, binData_, "pho");
  //	new EgHLTDQMVarCut<OffPho>(cutMasks_.stdPho,&OffPho::cutCode));
  //monElemFuncs.initTightLooseTrigHistsTrigCuts(phoMonElems_,phoTightLooseTrigNames_,binData_);

  //di-object triggers
  monElemFuncs.initTightLooseTrigHists(eleMonElems_, diEleTightLooseTrigNames_, binData_, "gsfEle");
  //	new EgDiEleCut(cutMasks_.stdEle,&OffEle::cutCode));
  monElemFuncs.initTightLooseTrigHists(phoMonElems_, diPhoTightLooseTrigNames_, binData_, "pho");
  //				new EgDiPhoCut(cutMasks_.stdPho,&OffPho::cutCode));

  monElemFuncs.initTightLooseDiObjTrigHistsTrigCuts(eleMonElems_, diEleTightLooseTrigNames_, binData_);
  monElemFuncs.initTightLooseDiObjTrigHistsTrigCuts(phoMonElems_, diPhoTightLooseTrigNames_, binData_);

  //tag and probe trigger efficiencies
  //this is to do measure the trigger efficiency with respect to a fully selected offline electron
  //using a tag and probe technique (note: this will be different to the trigger efficiency normally calculated)
  bool doTrigTagProbeEff = false;
  if (doTrigTagProbeEff && (!dohep_)) {
    for (auto const& eleHLTFilterName : eleHLTFilterNames_) {
      iBooker.setCurrentFolder(dirName_ + "/" + subdirName_ + "/" + eleHLTFilterName);
      monElemFuncs.initTrigTagProbeHist(eleMonElems_, eleHLTFilterName, cutMasks_.trigTPEle, binData_);
    }
    for (auto const& phoHLTFilterName : phoHLTFilterNames_) {
      iBooker.setCurrentFolder(dirName_ + "/" + subdirName_ + "/" + phoHLTFilterName);
      monElemFuncs.initTrigTagProbeHist(phoMonElems_, phoHLTFilterName, cutMasks_.trigTPPho, binData_);
    }
    for (auto& i : eleHLTFilterNames2Leg_) {
      iBooker.setCurrentFolder(dirName_ + "/" + subdirName_ + "/" + i.substr(i.find("::") + 2));
      //std::cout<<"FilterName: "<<eleHLTFilterNames2Leg_[i]<<std::endl;
      //std::cout<<"Folder: "<<eleHLTFilterNames2Leg_[i].substr(eleHLTFilterNames2Leg_[i].find("::")+2)<<std::endl;
      monElemFuncs.initTrigTagProbeHist_2Leg(eleMonElems_, i, cutMasks_.trigTPEle, binData_);
    }
    //tag and probe not yet implimented for photons (attemping to see if it makes sense first)
    // monElemFuncs.initTrigTagProbeHists(phoMonElems,phoHLTFilterNames_);
  }

  iBooker.setCurrentFolder(dirName_);
}

void EgHLTOfflineSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const double weight = 1.;  //we have the ability to weight but its disabled for now - maybe use this for prescales?
  nrEventsProcessed_++;
  nrEventsProcessedMonElem_->Fill(nrEventsProcessed_);
  int errCode = offEvtHelper_.makeOffEvt(iEvent, iSetup, offEvt_, *trigCodes);
  if (errCode != 0) {
    dqmErrsMonElem_->Fill(errCode);
    return;
  }

  for (auto& eleFilterMonHist : eleFilterMonHists_) {
    eleFilterMonHist->fill(offEvt_, weight);
  }
  for (auto& phoFilterMonHist : phoFilterMonHists_) {
    phoFilterMonHist->fill(offEvt_, weight);
  }

  for (auto& eleMonElem : eleMonElems_) {
    const std::vector<OffEle>& eles = offEvt_.eles();
    for (auto const& ele : eles) {
      eleMonElem->fill(ele, offEvt_, weight);
    }
  }

  for (auto& phoMonElem : phoMonElems_) {
    const std::vector<OffPho>& phos = offEvt_.phos();
    for (auto const& pho : phos) {
      phoMonElem->fill(pho, offEvt_, weight);
    }
  }
}

void EgHLTOfflineSource::addEleTrigPath(MonElemFuncs& monElemFuncs, const std::string& name) {
  auto* filterMon =
      new EleHLTFilterMon(monElemFuncs, name, trigCodes->getCode(name.c_str()), binData_, cutMasks_, dohep_);
  eleFilterMonHists_.push_back(filterMon);
  std::sort(eleFilterMonHists_.begin(), eleFilterMonHists_.end(), [](auto const& x, auto const& y) { return *x < *y; });
  //takes a minor efficiency hit at initalisation to ensure that the vector is always sorted
}

void EgHLTOfflineSource::addPhoTrigPath(MonElemFuncs& monElemFuncs, const std::string& name) {
  PhoHLTFilterMon* filterMon =
      new PhoHLTFilterMon(monElemFuncs, name, trigCodes->getCode(name.c_str()), binData_, cutMasks_, dohep_);
  phoFilterMonHists_.push_back(filterMon);
  std::sort(phoFilterMonHists_.begin(), phoFilterMonHists_.end(), [](auto const& x, auto const& y) { return *x < *y; });
  //takes a minor efficiency hit at initalisation to ensure that the vector is always sorted
}

//this function puts every filter name used in a std::vector
//due to the design, to ensure we get every filter, filters will be inserted multiple times
//eg electron filters will contain photon triggers which are also in the photon filters
//but only want one copy in the vector
//this function is intended to be called once per job so some inefficiency can can be tolerated
//therefore we will use a std::set to ensure that each filtername is only inserted once
//and then convert to a std::vector
void EgHLTOfflineSource::getHLTFilterNamesUsed(std::vector<std::string>& filterNames) const {
  std::set<std::string> filterNameSet;
  for (auto const& eleHLTFilterName : eleHLTFilterNames_)
    filterNameSet.insert(eleHLTFilterName);
  for (auto const& phoHLTFilterName : phoHLTFilterNames_)
    filterNameSet.insert(phoHLTFilterName);
  //here we are little more complicated as entries are of the form "tightTrig:looseTrig"
  //so we need to split them first
  for (auto const& eleTightLooseTrigName : eleTightLooseTrigNames_) {
    std::vector<std::string> trigNames;
    boost::split(trigNames, eleTightLooseTrigName, boost::is_any_of(std::string(":")));
    if (trigNames.size() != 2)
      continue;  //format incorrect
    filterNameSet.insert(trigNames[0]);
    filterNameSet.insert(trigNames[1]);
  }
  for (auto const& diEleTightLooseTrigName : diEleTightLooseTrigNames_) {
    std::vector<std::string> trigNames;
    boost::split(trigNames, diEleTightLooseTrigName, boost::is_any_of(std::string(":")));
    if (trigNames.size() != 2)
      continue;  //format incorrect
    filterNameSet.insert(trigNames[0]);
    filterNameSet.insert(trigNames[1]);
  }
  for (auto const& phoTightLooseTrigName : phoTightLooseTrigNames_) {
    std::vector<std::string> trigNames;
    boost::split(trigNames, phoTightLooseTrigName, boost::is_any_of(std::string(":")));
    if (trigNames.size() != 2)
      continue;  //format incorrect
    filterNameSet.insert(trigNames[0]);
    filterNameSet.insert(trigNames[1]);
  }
  for (auto const& diPhoTightLooseTrigName : diPhoTightLooseTrigNames_) {
    std::vector<std::string> trigNames;
    boost::split(trigNames, diPhoTightLooseTrigName, boost::is_any_of(std::string(":")));
    if (trigNames.size() != 2)
      continue;  //format incorrect
    filterNameSet.insert(trigNames[0]);
    filterNameSet.insert(trigNames[1]);
  }
  //right all the triggers are inserted once and only once in the set, convert to vector
  //very lazy, create a new vector so can use the constructor and then use swap to transfer
  std::vector<std::string>(filterNameSet.begin(), filterNameSet.end()).swap(filterNames);
}

void EgHLTOfflineSource::filterTriggers(const HLTConfigProvider& hltConfig) {
  std::vector<std::string> activeFilters;
  std::vector<std::string> activeEleFilters;
  std::vector<std::string> activeEle2LegFilters;
  std::vector<std::string> activePhoFilters;
  std::vector<std::string> activePho2LegFilters;

  trigTools::getActiveFilters(
      hltConfig, activeFilters, activeEleFilters, activeEle2LegFilters, activePhoFilters, activePho2LegFilters);

  trigTools::filterInactiveTriggers(eleHLTFilterNames_, activeEleFilters);
  trigTools::filterInactiveTriggers(phoHLTFilterNames_, activePhoFilters);
  trigTools::filterInactiveTriggers(eleHLTFilterNames2Leg_, activeEle2LegFilters);
  trigTools::filterInactiveTightLooseTriggers(eleTightLooseTrigNames_, activeEleFilters);
  trigTools::filterInactiveTightLooseTriggers(diEleTightLooseTrigNames_, activeEleFilters);
  trigTools::filterInactiveTightLooseTriggers(phoTightLooseTrigNames_, activePhoFilters);
  trigTools::filterInactiveTightLooseTriggers(diPhoTightLooseTrigNames_, activePhoFilters);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgHLTOfflineSource);
