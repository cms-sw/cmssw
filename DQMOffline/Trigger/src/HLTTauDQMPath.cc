#include "DQMOffline/Trigger/interface/HLTTauDQMPath.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Math/interface/deltaR.h"

#include<cstdio>
#include<sstream>
#include<algorithm>

namespace {
  // Used as a helper only in this file
  class HLTPath {
  public:
    HLTPath(const std::string& name, const boost::smatch& what):
      name_(name)
    {
      char buffer[10];
      for(int i=0; i<10; ++i) {
        snprintf(buffer, 10, "tr%d", i);
        boost::ssub_match sm = what[buffer];
        if(sm.length() == 0)
          break;

        try {
          thresholds_.push_back(std::stoi(sm.str())); // C++11
        } catch(std::invalid_argument& e) {
          throw cms::Exception("Configuration") << "Interpreting regex of path " << name << ", threshold " << buffer << ": unable to convert '" << sm.str() << "' to integer";
        } catch(std::out_of_range& e) {
          throw cms::Exception("Configuration") << "Interpreting regex of path " << name << ", threshold " << buffer << ": '" << sm.str() << "' is out of int range";
        }
      }
    }

    bool isBetterThan(const HLTPath& other, const HLTConfigProvider& HLTCP)  const {
      // First compare prescales
      // Search for prescale set where either is enabled
      // If other is disabled (prescale = 0), pick the enabled one
      // If prescales are different, pick the one with smaller
      // If prescales are same, continue to compare thresholds
      for(unsigned int iSet = 0; iSet < HLTCP.prescaleSize(); ++iSet) {
        unsigned int prescale = HLTCP.prescaleValue(iSet, name_);
        unsigned int prescaleOther = HLTCP.prescaleValue(iSet, other.name_);
        if(prescale == 0 && prescaleOther == 0)
          continue;
        if(prescale == 0)
          return false;
        if(prescaleOther == 0)
          return true;

        if(prescale != prescaleOther)
          return prescale < prescaleOther;
        break;
      }

      // Then thresholds
      if(thresholds_.size() != other.thresholds_.size())
        throw cms::Exception("Configuration") << "Comparing path " << name_ << " and " << other.name_ << ", they have different numbers of thresholds (" << thresholds_.size() << " != " << other.thresholds_.size() << ")";
      for(size_t i=0; i<thresholds_.size(); ++i) {
        if(thresholds_[i] != other.thresholds_[i])
          return thresholds_[i] < other.thresholds_[i];
      }

      // Nothing left, what to do? In principle this is an error
      throw cms::Exception("Configuration") << "Comparing path " << name_ << " and " << other.name_ << ", unable to tell which to choose. Improve your input regex!";
      return true;
    }

    typedef HLTTauDQMPath::FilterIndex FilterIndex;

    std::vector<FilterIndex> interestingFilters(const HLTConfigProvider& HLTCP, bool doRefAnalysis, const std::vector<boost::regex>& ignoreFilterTypes, const std::vector<boost::regex>& ignoreFilterNames) const {
      const std::vector<std::string>& moduleLabels = HLTCP.moduleLabels(name_);
      std::vector<FilterIndex> selectedFilters;
      std::vector<std::string> leptonTauFilters;

      // Ignore all "Selector"s, for ref-analysis keep only those with saveTags=True
      // Also record HLT2(Electron|Muon)(PF)?Tau module names
      for(std::vector<std::string>::const_iterator iLabel = moduleLabels.begin(); iLabel != moduleLabels.end(); ++iLabel) {
        if(HLTCP.moduleEDMType(*iLabel) != "EDFilter")
          continue;
        const std::string type = HLTCP.moduleType(*iLabel);
        if(type.find("Selector") != std::string::npos)
          continue;
        if(type == "HLTTriggerTypeFilter" || type == "HLTBool")
          continue;
        if(doRefAnalysis && !HLTCP.saveTags(*iLabel))
          continue;
        if(type == "HLT2ElectronPFTau" || type == "HLT2MuonPFTau" || type == "HLT2ElectronTau" || type == "HLT2MuonTau")
          leptonTauFilters.emplace_back(*iLabel);
        else if(type.find("Electron") != std::string::npos || type.find("Egamma") != std::string::npos || type.find("Muon") != std::string::npos)
          continue;
        selectedFilters.emplace_back(*iLabel, iLabel-moduleLabels.begin());
      }

      // Insert the last filters of lepton legs
      for(const std::string& leptonTauLabel: leptonTauFilters) {
        const edm::ParameterSet& pset = HLTCP.modulePSet(leptonTauLabel);
        std::string input1 = pset.getParameter<edm::InputTag>("inputTag1").label();
        std::string input2 = pset.getParameter<edm::InputTag>("inputTag2").label();
        unsigned idx1 = HLTCP.moduleIndex(name_, input1);
        unsigned idx2 = HLTCP.moduleIndex(name_, input2);

        auto func = [&](const FilterIndex& a, unsigned idxb) {
          return std::get<1>(a) < idxb;
        };
        std::vector<FilterIndex>::iterator found = std::lower_bound(selectedFilters.begin(), selectedFilters.end(), idx1, func);
        if(found == selectedFilters.end() || std::get<1>(*found) != idx1)
          selectedFilters.emplace(found, input1, idx1);
        found = std::lower_bound(selectedFilters.begin(), selectedFilters.end(), idx2, func);
        if(found == selectedFilters.end() || std::get<1>(*found) != idx2)
          selectedFilters.emplace(found, input2, idx2);
      }

      // Remove filters ignored by their type
      std::vector<FilterIndex>::iterator selectedFiltersEnd = std::remove_if(selectedFilters.begin(), selectedFilters.end(), [&](const FilterIndex& labelIndex) {
          for(const boost::regex& re: ignoreFilterTypes) {
            if(boost::regex_search(HLTCP.moduleType(std::get<0>(labelIndex)), re))
              return true;
          }
          return false;
        });
      // Remove filters ignored by their label
      selectedFiltersEnd = std::remove_if(selectedFilters.begin(), selectedFiltersEnd, [&](const FilterIndex& labelIndex) {
          for(const boost::regex& re: ignoreFilterNames) {
            if(boost::regex_search(std::get<0>(labelIndex), re))
              return true;
          }
          return false;
        });


      std::vector<FilterIndex> ret;
      ret.reserve(selectedFiltersEnd-selectedFilters.begin());
      std::move(selectedFilters.begin(), selectedFiltersEnd, std::back_inserter(ret));
      return ret;
    }

    size_t tauProducerIndex(const HLTConfigProvider& HLTCP) const {
      const std::vector<std::string>& moduleLabels = HLTCP.moduleLabels(name_);
      for(std::vector<std::string>::const_iterator iLabel = moduleLabels.begin(); iLabel != moduleLabels.end(); ++iLabel) {
        const std::string type = HLTCP.moduleType(*iLabel);
        if(type == "PFRecoTauProducer") {
          //edm::LogInfo("HLTTauDQMOffline") << "Found PFTauProducer " << *iLabel << " index " << (iLabel-moduleLabels.begin());
          return iLabel-moduleLabels.begin();
        }
      }
      return std::numeric_limits<size_t>::max();
    }

    const std::string& name() const { return name_; }

  private:
    std::string name_;

    std::vector<int> thresholds_;
  };

  int getParameterSafe(const HLTConfigProvider& HLTCP, const std::string& filterName, const std::string& parameterName) {
    const edm::ParameterSet& pset = HLTCP.modulePSet(filterName);
    if(pset.existsAs<int>(parameterName))
      return pset.getParameter<int>(parameterName);
    else {
      edm::LogWarning("HLTTauDQMOfflineSource") << "No parameter '" << parameterName << "' in configuration of filter " << filterName << " pset " << pset.dump() << std::endl;
      return 0;
    }
  }

  struct TauLeptonMultiplicity {
    TauLeptonMultiplicity(): tau(0), electron(0), muon(0) {}
    int tau;
    int electron;
    int muon;
  };
  TauLeptonMultiplicity inferTauLeptonMultiplicity(const HLTConfigProvider& HLTCP, const std::string& filterName, const std::string& moduleType) {
    TauLeptonMultiplicity n;

    if(moduleType == "HLTLevel1GTSeed") {
      if(filterName.find("SingleMu") != std::string::npos) {
        n.muon = 1;
      }
      else if(filterName.find("SingleEG") != std::string::npos) {
        n.electron = 1;
      }
      else if(filterName.find("DoubleTau") != std::string::npos) {
        n.tau = 2;
      }
    }
    else if(moduleType == "HLT1CaloJet") {
      //const edm::ParameterSet& pset = HLTCP.modulePSet(filterName);
      //pset.getParameter<int>("triggerType") == trigger::TriggerTau) {
      if(getParameterSafe(HLTCP, filterName, "triggerType") == trigger::TriggerTau) {
        //n.tau = pset.getParameter<int>("MinN");
        n.tau = getParameterSafe(HLTCP, filterName, "MinN");
      }
    }
    else if(moduleType == "HLTCaloJetTag") {
      //const edm::ParameterSet& pset = HLTCP.modulePSet(filterName);
      //if(pset.getParameter<int>("triggerType") == trigger::TriggerTau) {
      if(getParameterSafe(HLTCP, filterName, "TriggerType") == trigger::TriggerTau) {
        //n.tau = pset.getParameter<int>("MinJets");
        n.tau = getParameterSafe(HLTCP, filterName, "MinJets");
      }
    }
    else if(moduleType == "HLT1PFTau") {
      //n.tau = HLTCP.modulePSet(filterName).getParameter<int>("MinN");
      n.tau = getParameterSafe(HLTCP, filterName, "MinN");
    }
    else if(moduleType == "HLTPFTauPairDzMatchFilter") {
      n.tau = 2;
    }
    else if(moduleType == "HLTElectronGenericFilter") {
      //n.electron = HLTCP.modulePSet(filterName).getParameter<int>("ncandcut");
      n.electron = getParameterSafe(HLTCP, filterName, "ncandcut");
    }
    else if(moduleType == "HLTMuonIsoFilter") {
      n.muon = HLTCP.modulePSet(filterName).getParameter<int>("MinN");
    }
    else if(moduleType == "HLT2ElectronTau" || moduleType == "HLT2ElectronPFTau") {
      //int num = HLTCP.modulePSet(filterName).getParameter<int>("MinN");
      int num = getParameterSafe(HLTCP, filterName, "MinN");
      n.tau = num;
      n.electron = num;
    }
    else if(moduleType == "HLT2MuonPFTau") {
      //int num = HLTCP.modulePSet(filterName).getParameter<int>("MinN");
      int num = getParameterSafe(HLTCP, filterName, "MinN");
      n.tau = num;
      n.muon = num;
    }
    else if(moduleType == "HLTPrescaler" || moduleType == "HLT1CaloMET") {
      // ignore
    }
    else {
      edm::LogWarning("HLTTauDQMOfflineSource") << "HLTTauDQMPath.cc, inferTauLeptonMultiplicity(): module type '" << moduleType << "' not recognized, filter '" << filterName << "' will be ignored for offline matching." << std::endl;
    }

    return n;
  }

  template <typename T1, typename T2>
  bool deltaRmatch(const T1& obj, const std::vector<T2>& refColl, double dR, std::vector<T2>& matchedRefs) {
    double minDr = 2*dR;
    size_t found = refColl.size();
    //std::cout << "Matching with DR " << dR << ", obj eta " << obj.eta() << " phi " << obj.phi() << std::endl;
    for(size_t i=0; i<refColl.size(); ++i) {
      double dr = reco::deltaR(obj, refColl[i]);
      //std::cout << "  " << i << " ref eta " << refColl[i].eta() << " phi " << refColl[i].phi() << " dr " << dr << std::endl;
      if(dr < minDr) {
        minDr = dr;
        found = i;
      }
    }
    if(found < refColl.size()) {
      bool matchedAlreadyIn = false;
      for(const T2& mobj: matchedRefs) {
        if(reco::deltaR(refColl[found], mobj) < 0.0001) {
          matchedAlreadyIn = true;
          break;
        }
      }
      if(!matchedAlreadyIn)
        matchedRefs.emplace_back(refColl[found]);
      return true;
    }
    return false;
  }
}


HLTTauDQMPath::HLTTauDQMPath(const std::string& hltProcess, bool doRefAnalysis):
  hltProcess_(hltProcess),
  doRefAnalysis_(doRefAnalysis),
  pathIndex_(0),
  lastFilterBeforeL2TauIndex_(0), lastL2TauFilterIndex_(0),
  lastFilterBeforeL3TauIndex_(0), lastL3TauFilterIndex_(0),
  isFirstL1Seed_(false)
{}
HLTTauDQMPath::~HLTTauDQMPath() {}

void HLTTauDQMPath::initialize(const edm::ParameterSet& pset) {
  std::vector<std::string> regexs;
  std::vector<boost::regex> ignoreFilterTypes;
  std::vector<boost::regex> ignoreFilterNames;
  std::vector<std::string> regexsTmp = pset.getUntrackedParameter<std::vector<std::string> >("Path");
  pathRegexs_.reserve(regexsTmp.size());
  for(const std::string& str: regexsTmp)
    pathRegexs_.emplace_back(str);

  std::vector<std::string> ignoreFilterTypesTmp;
  std::vector<std::string> ignoreFilterNamesTmp;
  ignoreFilterTypesTmp   = pset.getUntrackedParameter<std::vector<std::string> >("IgnoreFilterTypes");
  ignoreFilterNamesTmp   = pset.getUntrackedParameter<std::vector<std::string> >("IgnoreFilterNames");
  ignoreFilterTypes.reserve(ignoreFilterTypesTmp.size());
  ignoreFilterNames.reserve(ignoreFilterNamesTmp.size());
  for(const std::string& str: ignoreFilterTypesTmp)
    ignoreFilterTypes.emplace_back(str);
  for(const std::string& str: ignoreFilterNamesTmp)
    ignoreFilterNames.emplace_back(str);
}

bool HLTTauDQMPath::beginRun(const HLTConfigProvider& HLTCP) {
  // Search path candidates

  std::vector<HLTPath> foundPaths;
  const std::vector<std::string>& triggerNames = HLTCP.triggerNames();
  for(const boost::regex& re: pathRegexs_) {
    //std::cout << regexStr << std::endl;
    boost::smatch what;

    for(const std::string& path: triggerNames) {
      if(boost::regex_match(path, what, re)) {
        foundPaths.emplace_back(path, what);
      }
    }
    if(!foundPaths.empty())
      break;
  }
  if(foundPaths.empty()) {
    std::stringstream ss;
    for(std::vector<boost::regex>::const_iterator iRegex = pathRegexs_.begin(); iRegex != pathRegexs_.end(); ++iRegex) {
      if(iRegex != pathRegexs_.begin())
        ss << ",";
      ss << iRegex->str();
    }
    edm::LogWarning("HLTTauDQMOffline") << "HLTTauDQMPath::beginRun(): did not find any paths matching to regexes " << ss.str();
    return false;
  }

  // If more than one, find the best match
  std::vector<HLTPath>::const_iterator thePath = foundPaths.begin();
  std::vector<HLTPath>::const_iterator iPath = thePath;
  ++iPath;
  for(; iPath != foundPaths.end(); ++iPath) {
    if(!thePath->isBetterThan(*iPath, HLTCP))
      thePath = iPath;
  }
  std::stringstream ss;
  ss << "HLTTauDQMPath::beginRun(): Chose path " << thePath->name() << "\n";

  // Get the filters
  filterIndices_ = thePath->interestingFilters(HLTCP, doRefAnalysis_, ignoreFilterTypes_, ignoreFilterNames_);
  isFirstL1Seed_ = HLTCP.moduleType(std::get<0>(filterIndices_[0])) == "HLTLevel1GTSeed";
  ss << "  Filters";
  // Set the filter multiplicity counts
  filterTauN_.clear();
  filterElectronN_.clear();
  filterMuonN_.clear();
  filterTauN_.reserve(filterIndices_.size());
  filterElectronN_.reserve(filterIndices_.size());
  filterMuonN_.reserve(filterIndices_.size());
  for(size_t i=0; i<filterIndices_.size(); ++i) {
    const std::string& filterName = std::get<0>(filterIndices_[i]);
    const std::string& moduleType = HLTCP.moduleType(filterName);

    TauLeptonMultiplicity n = inferTauLeptonMultiplicity(HLTCP, filterName, moduleType);
    filterTauN_.push_back(n.tau);
    filterElectronN_.push_back(n.electron);
    filterMuonN_.push_back(n.muon);

    ss << "\n    " << std::get<1>(filterIndices_[i])
       << " " << filterName
       << " " << moduleType
       << " ntau " << n.tau
       << " nele " << n.electron
       << " nmu " << n.muon;

  }
  edm::LogInfo("HLTTauDQMOffline") << ss.str();


  // Find the position of PFRecoTauProducer, use filters with taus
  // before it for L2 tau efficiency, and filters with taus after it
  // for L3 tau efficiency
  const size_t tauProducerIndex = thePath->tauProducerIndex(HLTCP);
  if(tauProducerIndex == std::numeric_limits<size_t>::max()) {
    edm::LogWarning("HLTTauDQMOffline") << "HLTTauDQMPath::beginRun(): Did not find PFRecoTauProducer from HLT path " << thePath->name();
    return false;
  }
  //lastFilterBeforeL2TauIndex_ = std::numeric_limits<size_t>::max();
  lastL2TauFilterIndex_ = std::numeric_limits<size_t>::max();
  //lastFilterBeforeL3TauIndex_ = std::numeric_limits<size_t>::max();
  lastL3TauFilterIndex_ = std::numeric_limits<size_t>::max();
  size_t i = 0;
  lastFilterBeforeL2TauIndex_ = 0;
  for(; i<filtersSize() && getFilterIndex(i) < tauProducerIndex; ++i) {
    if(lastL2TauFilterIndex_ == std::numeric_limits<size_t>::max() && getFilterNTaus(i) == 0)
      lastFilterBeforeL2TauIndex_ = i;
    if(getFilterNTaus(i) > 0 && getFilterNElectrons(i) == 0 && getFilterNMuons(i) == 0)
      lastL2TauFilterIndex_ = i;
  }
  lastFilterBeforeL3TauIndex_ = i-1;
  for(; i<filtersSize(); ++i) {
    if(lastL3TauFilterIndex_ == std::numeric_limits<size_t>::max() && getFilterNTaus(i) == 0)
      lastFilterBeforeL3TauIndex_ = i;
    if(getFilterNTaus(i) > 0 && getFilterNElectrons(i) == 0 && getFilterNMuons(i) == 0)
      lastL3TauFilterIndex_ = i;
  }
  edm::LogInfo("HLTTauDQMOffline") << "lastFilterBeforeL2 " << lastFilterBeforeL2TauIndex_
                                   << " lastL2TauFilter " << lastL2TauFilterIndex_
                                   << " lastFilterBeforeL3 " << lastFilterBeforeL3TauIndex_
                                   << " lastL3TauFilter " << lastL3TauFilterIndex_;

  // Set path index
  pathName_ = thePath->name();
  pathIndex_ = HLTCP.triggerIndex(thePath->name());


  return true;
}

bool HLTTauDQMPath::fired(const edm::TriggerResults& triggerResults) const {
  return triggerResults.accept(pathIndex_);
}

int HLTTauDQMPath::lastPassedFilter(const edm::TriggerResults& triggerResults) const {
  if(fired(triggerResults)) {
    //std::cout << "Event passed" << std::endl;
    return filterIndices_.size()-1;
  }

  unsigned int firstFailedFilter = triggerResults.index(pathIndex_);
  int lastPassedFilter = -1;
  for(size_t i=0; i<filterIndices_.size(); ++i) {
    if(std::get<1>(filterIndices_[i]) < firstFailedFilter) {
      lastPassedFilter = i;
    }
    else {
      //std::cout << "Decision-making filter " << firstFailedFilter << " this " << std::get<1>(filterIndices_[i]) << std::endl;
      break;
    }
  }
  return lastPassedFilter;
}

void HLTTauDQMPath::getFilterObjects(const trigger::TriggerEvent& triggerEvent, size_t i, std::vector<Object>& retval) const {
  trigger::size_type filterIndex = triggerEvent.filterIndex(edm::InputTag(getFilterName(i), "", hltProcess_));
  if(filterIndex != triggerEvent.sizeFilters()) {
    const trigger::Keys& keys = triggerEvent.filterKeys(filterIndex);
    const trigger::Vids& ids = triggerEvent.filterIds(filterIndex);
    const trigger::TriggerObjectCollection& triggerObjects = triggerEvent.getObjects();
    //std::cout << "Filter name " << getFilterName(i) << std::endl;
    for(size_t i=0; i<keys.size(); ++i) {
      const trigger::TriggerObject& object = triggerObjects[keys[i]];
      retval.emplace_back(Object{object, ids[i]});
      //std::cout << "  object id " <<  object.id() << std::endl;
    }
  }
}

bool HLTTauDQMPath::offlineMatching(size_t i, const std::vector<Object>& triggerObjects, const HLTTauDQMOfflineObjects& offlineObjects, double dR, std::vector<Object>& matchedTriggerObjects, HLTTauDQMOfflineObjects& matchedOfflineObjects) const {
  bool isL1 = (i==0 && isFirstL1Seed_);
  if(filterTauN_[i] > 0) {
    int matchedObjects = 0;
    for(const Object& trgObj: triggerObjects) {
      //std::cout << "trigger object id " << trgObj.id << std::endl;
      if(! ((isL1 && (trgObj.id == trigger::TriggerL1TauJet || trgObj.id == trigger::TriggerL1CenJet))
            || trgObj.id == trigger::TriggerTau) )
        continue;
      if(deltaRmatch(trgObj.object, offlineObjects.taus, dR, matchedOfflineObjects.taus)) {
        ++matchedObjects;
        matchedTriggerObjects.emplace_back(trgObj);
      }
    }
    if(matchedObjects < filterTauN_[i])
      return false;
  }
  if(filterElectronN_[i] > 0) {
    int matchedObjects = 0;
    for(const Object& trgObj: triggerObjects) {
      //std::cout << "trigger object id " << trgObj.id << std::endl;
      if(! ((isL1 && (trgObj.id == trigger::TriggerL1NoIsoEG || trgObj.id == trigger::TriggerL1IsoEG))
            || trgObj.id == trigger::TriggerElectron) )
        continue;
      if(deltaRmatch(trgObj.object, offlineObjects.electrons, dR, matchedOfflineObjects.electrons)) {
        ++matchedObjects;
        matchedTriggerObjects.emplace_back(trgObj);
      }
    }
    if(matchedObjects < filterElectronN_[i])
      return false;
  }
  if(filterMuonN_[i] > 0) {
    int matchedObjects = 0;
    for(const Object& trgObj: triggerObjects) {
      //std::cout << "trigger object id " << trgObj.id << std::endl;
      if(! ((isL1 && trgObj.id == trigger::TriggerL1Mu)
            || trgObj.id == trigger::TriggerMuon) )
        continue;
      if(deltaRmatch(trgObj.object, offlineObjects.muons, dR, matchedOfflineObjects.muons)) {
        ++matchedObjects;
        matchedTriggerObjects.emplace_back(trgObj);
      }
    }
    if(matchedObjects < filterMuonN_[i])
      return false;
  }
  // Sort offline objects by pt
  std::sort(matchedOfflineObjects.taus.begin(), matchedOfflineObjects.taus.end(), [](const LV& a, const LV&b) { return a.pt() > b.pt();});
  std::sort(matchedOfflineObjects.electrons.begin(), matchedOfflineObjects.electrons.end(), [](const LV& a, const LV&b) { return a.pt() > b.pt();});
  std::sort(matchedOfflineObjects.muons.begin(), matchedOfflineObjects.muons.end(), [](const LV& a, const LV&b) { return a.pt() > b.pt();});

  return true;
}

bool HLTTauDQMPath::goodOfflineEvent(size_t i, const HLTTauDQMOfflineObjects& offlineObjects) const {
  return (static_cast<size_t>(getFilterNTaus(i)) <= offlineObjects.taus.size() &&
          static_cast<size_t>(getFilterNElectrons(i)) <= offlineObjects.electrons.size() &&
          static_cast<size_t>(getFilterNMuons(i)) <= offlineObjects.muons.size());
}
