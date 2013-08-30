// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMPath_h
#define DQMOffline_Trigger_HLTTauDQMPath_h

#include "DataFormats/Math/interface/LorentzVector.h"

#include <boost/regex.hpp>

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

class HLTTauDQMPath {
public:
  typedef math::XYZTLorentzVectorD LV;
  typedef std::vector<LV> LVColl;

  struct Object {
    const trigger::TriggerObject& object;
    const int id; // from TriggerTypeDefs.h
  };

  HLTTauDQMPath(const std::string& hltProcess, bool doRefAnalysis);
  ~HLTTauDQMPath();

  void initialize(const edm::ParameterSet& pset);

  bool beginRun(const HLTConfigProvider& HLTCP);

  bool fired(const edm::TriggerResults& triggerResults) const;

  // index (to getFilterName, getFilterIndex) of the last passed filter
  // -1, if the first filter rejects the event
  int lastPassedFilter(const edm::TriggerResults& triggerResults) const;

  const std::string& getPathName() const { return pathName_; }
  const unsigned int getPathIndex() const { return pathIndex_; }

  size_t filtersSize() const { return filterIndices_.size(); }
  const std::string& getFilterName(size_t i) const { return std::get<0>(filterIndices_[i]); }

  bool isFirstFilterL1Seed() const { return isFirstL1Seed_; }
  const std::string& getLastFilterName() const { return std::get<0>(filterIndices_.back()); }

  // index (to edm::TriggerResults) of a filter
  size_t getFilterIndex(size_t i) const { return std::get<1>(filterIndices_[i]); }

  // Get objects associated to a filter, i is the "internal" index
  void getFilterObjects(const trigger::TriggerEvent& triggerEvent, size_t i, std::vector<Object>& retval) const;

  // i = filter index
  bool offlineMatching(size_t i, const std::vector<Object>& triggerObjects, const std::map<int, LVColl>& offlineObjects, double dR, std::vector<Object>& matchedTriggerObjects, LVColl& matchedOfflineObjects) const;

  typedef std::tuple<std::string, size_t> FilterIndex;
private:
  const std::string hltProcess_;
  const bool doRefAnalysis_;

  std::vector<boost::regex> pathRegexs_;
  std::vector<boost::regex> ignoreFilterTypes_;
  std::vector<boost::regex> ignoreFilterNames_;

  std::vector<FilterIndex> filterIndices_;
  std::vector<int> filterTauN_;
  std::vector<int> filterLeptonN_;
  std::string pathName_;
  unsigned int pathIndex_;
  bool isFirstL1Seed_;
};


#endif
