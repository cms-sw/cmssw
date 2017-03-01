// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMPath_h
#define DQMOffline_Trigger_HLTTauDQMPath_h

#include "DataFormats/Math/interface/LorentzVector.h"

#include <tuple>
#include <vector>
#include <string>

class HLTConfigProvider;
namespace edm {
  class ParameterSet;
  class TriggerResults;
}
namespace trigger {
  class TriggerEvent;
  class TriggerObject;
}
struct HLTTauDQMOfflineObjects;

class HLTTauDQMPath {
public:
  typedef math::XYZTLorentzVectorD LV;
  typedef std::vector<LV> LVColl;
  typedef std::tuple<std::string, std::string, size_t> FilterIndex;

  constexpr static size_t kName = 0;
  constexpr static size_t kType = 1;
  constexpr static size_t kModuleIndex = 2;
  constexpr static size_t kInvalidIndex = std::numeric_limits<size_t>::max();

  struct Object {
    const trigger::TriggerObject& object;
    const int id; // from TriggerTypeDefs.h
  };

  HLTTauDQMPath(const std::string& pathName, const std::string& hltProcess, bool doRefAnalysis, const HLTConfigProvider& HLTCP);
  ~HLTTauDQMPath();

  bool isValid() const { return isValid_; }

  bool fired(const edm::TriggerResults& triggerResults) const;

  // index (to getFilterName, getFilterIndex) of the last passed filter
  // -1, if the first filter rejects the event
  int lastPassedFilter(const edm::TriggerResults& triggerResults) const;

  const std::string& getPathName() const { return pathName_; }
  const unsigned int getPathIndex() const { return pathIndex_; }

  size_t filtersSize() const { return filterIndices_.size(); }
  const std::string& getFilterName(size_t i) const { return std::get<kName>(filterIndices_[i]); }
  const std::string& getFilterType(size_t i) const { return std::get<kType>(filterIndices_[i]); }
  int getFilterNTaus(size_t i) const { if(i < filterTauN_.size()) return filterTauN_[i]; else return 0;}
  int getFilterNElectrons(size_t i) const {if(i < filterElectronN_.size()) return filterElectronN_[i]; else return 0;}
  int getFilterNMuons(size_t i) const {if(i < filterMuonN_.size()) return filterMuonN_[i]; else return 0;}
  int getFilterMET(size_t i) const {if(i < filterMET_.size()) return filterMET_[i]; else return 0;}
  int getFilterLevel(size_t i) const {if(i < filterLevel_.size()) return filterLevel_[i]; else return 0;}

  bool isFirstFilterL1Seed() const { return isFirstL1Seed_; }
  const std::string& getLastFilterName() const { return std::get<kName>(filterIndices_.back()); }

  bool hasL2Taus() const { return lastL2TauFilterIndex_ != kInvalidIndex; }
  bool hasL3Taus() const { return lastL3TauFilterIndex_ != kInvalidIndex; }
  bool hasL2Electrons() const { return lastL2ElectronFilterIndex_ != kInvalidIndex; }  
  bool hasL3Electrons() const { return lastL3ElectronFilterIndex_ != kInvalidIndex; }  
  bool hasL2Muons() const { return lastL2MuonFilterIndex_ != kInvalidIndex; }
  bool hasL3Muons() const { return lastL3MuonFilterIndex_ != kInvalidIndex; }
  bool hasL2CaloMET() const { return lastL2METFilterIndex_ != kInvalidIndex; }
  size_t getLastFilterBeforeL2TauIndex() const { return lastFilterBeforeL2TauIndex_; }
  size_t getLastL2TauFilterIndex() const { return lastL2TauFilterIndex_; }
  size_t getLastFilterBeforeL3TauIndex() const { return lastFilterBeforeL3TauIndex_; }
  size_t getLastL3TauFilterIndex() const { return lastL3TauFilterIndex_; }

  size_t getLastFilterBeforeL2ElectronIndex() const { return lastFilterBeforeL2ElectronIndex_; }
  size_t getLastL2ElectronFilterIndex() const { return lastL2ElectronFilterIndex_; }
  size_t getLastFilterBeforeL3ElectronIndex() const { return lastFilterBeforeL3ElectronIndex_; }
  size_t getLastL3ElectronFilterIndex() const { return lastL3ElectronFilterIndex_; }

  size_t getLastFilterBeforeL2MuonIndex() const { return lastFilterBeforeL2MuonIndex_; }
  size_t getLastL2MuonFilterIndex() const { return lastL2MuonFilterIndex_; }
  size_t getLastFilterBeforeL3MuonIndex() const { return lastFilterBeforeL3MuonIndex_; }
  size_t getLastL3MuonFilterIndex() const { return lastL3MuonFilterIndex_; }

  size_t getLastFilterBeforeL2CaloMETIndex() const { return lastFilterBeforeL2METIndex_; }
  size_t getLastL2CaloMETFilterIndex() const { return lastL2METFilterIndex_; }
  size_t getFirstFilterBeforeL2CaloMETIndex() const { return firstFilterBeforeL2METIndex_; }
  size_t getFirstL2CaloMETFilterIndex() const { return firstL2METFilterIndex_; }

  // index (to edm::TriggerResults) of a filter
  size_t getFilterIndex(size_t i) const { return std::get<kModuleIndex>(filterIndices_[i]); }

  // Get objects associated to a filter, i is the "internal" index
  void getFilterObjects(const trigger::TriggerEvent& triggerEvent, size_t i, std::vector<Object>& retval) const;

  // i = filter index
  bool offlineMatching(size_t i, const std::vector<Object>& triggerObjects, const HLTTauDQMOfflineObjects& offlineObjects, double dR, std::vector<Object>& matchedTriggerObjects, HLTTauDQMOfflineObjects& matchedOfflineObjects) const;

  bool goodOfflineEvent(size_t i, const HLTTauDQMOfflineObjects& offlineObjects) const;

private:
  const std::string hltProcess_;
  const bool doRefAnalysis_;

  std::vector<FilterIndex> filterIndices_;
  std::vector<int> filterTauN_;
  std::vector<int> filterElectronN_;
  std::vector<int> filterMuonN_;
  std::vector<int> filterMET_;
  std::vector<int> filterLevel_;
  const std::string pathName_;
  const unsigned int pathIndex_;
  size_t lastFilterBeforeL2TauIndex_;
  size_t lastL2TauFilterIndex_;
  size_t lastFilterBeforeL3TauIndex_;
  size_t lastL3TauFilterIndex_;
  size_t lastFilterBeforeL2ElectronIndex_;
  size_t lastL2ElectronFilterIndex_;
  size_t lastFilterBeforeL3ElectronIndex_;
  size_t lastL3ElectronFilterIndex_;
  size_t lastFilterBeforeL2MuonIndex_;
  size_t lastL2MuonFilterIndex_;
  size_t lastFilterBeforeL3MuonIndex_;
  size_t lastL3MuonFilterIndex_;
  size_t lastFilterBeforeL2METIndex_;
  size_t lastL2METFilterIndex_;
  size_t firstFilterBeforeL2METIndex_;
  size_t firstL2METFilterIndex_;
  bool isFirstL1Seed_;
  bool isValid_;
};


#endif
