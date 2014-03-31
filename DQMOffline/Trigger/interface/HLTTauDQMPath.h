// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMPath_h
#define DQMOffline_Trigger_HLTTauDQMPath_h

#include "DataFormats/Math/interface/LorentzVector.h"

#include<tuple>
#include<vector>
#include<string>

class HLTConfigProvider;
namespace edm {
  class ParameterSet;
  class TriggerResults;
}
namespace trigger {
  class TriggerEvent;
  class TriggerObject;
}
class HLTTauDQMOfflineObjects;

class HLTTauDQMPath {
public:
  typedef math::XYZTLorentzVectorD LV;
  typedef std::vector<LV> LVColl;

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
  const std::string& getFilterName(size_t i) const { return std::get<0>(filterIndices_[i]); }
  int getFilterNTaus(size_t i) const { return filterTauN_[i]; }
  int getFilterNElectrons(size_t i) const {return filterElectronN_[i]; }
  int getFilterNMuons(size_t i) const {return filterMuonN_[i]; }

  bool isFirstFilterL1Seed() const { return isFirstL1Seed_; }
  const std::string& getLastFilterName() const { return std::get<0>(filterIndices_.back()); }

  bool hasL2Taus() const { return lastL2TauFilterIndex_ != std::numeric_limits<size_t>::max(); }
  bool hasL3Taus() const { return lastL3TauFilterIndex_ != std::numeric_limits<size_t>::max(); }
  size_t getLastFilterBeforeL2TauIndex() const { return lastFilterBeforeL2TauIndex_; }
  size_t getLastL2TauFilterIndex() const { return lastL2TauFilterIndex_; }
  size_t getLastFilterBeforeL3TauIndex() const { return lastFilterBeforeL3TauIndex_; }
  size_t getLastL3TauFilterIndex() const { return lastL3TauFilterIndex_; }

  // index (to edm::TriggerResults) of a filter
  size_t getFilterIndex(size_t i) const { return std::get<1>(filterIndices_[i]); }

  // Get objects associated to a filter, i is the "internal" index
  void getFilterObjects(const trigger::TriggerEvent& triggerEvent, size_t i, std::vector<Object>& retval) const;

  // i = filter index
  bool offlineMatching(size_t i, const std::vector<Object>& triggerObjects, const HLTTauDQMOfflineObjects& offlineObjects, double dR, std::vector<Object>& matchedTriggerObjects, HLTTauDQMOfflineObjects& matchedOfflineObjects) const;

  bool goodOfflineEvent(size_t i, const HLTTauDQMOfflineObjects& offlineObjects) const;

  typedef std::tuple<std::string, size_t> FilterIndex;
private:
  const std::string hltProcess_;
  const bool doRefAnalysis_;

  std::vector<FilterIndex> filterIndices_;
  std::vector<int> filterTauN_;
  std::vector<int> filterElectronN_;
  std::vector<int> filterMuonN_;
  const std::string pathName_;
  const unsigned int pathIndex_;
  size_t lastFilterBeforeL2TauIndex_;
  size_t lastL2TauFilterIndex_;
  size_t lastFilterBeforeL3TauIndex_;
  size_t lastL3TauFilterIndex_;
  bool isFirstL1Seed_;
  bool isValid_;
};


#endif
