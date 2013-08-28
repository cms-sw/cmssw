// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMPath_h
#define DQMOffline_Trigger_HLTTauDQMPath_h

#include <boost/regex.hpp>

#include<tuple>
#include<vector>

class HLTConfigProvider;
namespace edm {
  class ParameterSet;
  class TriggerResults;
}

class HLTTauDQMPath {
public:
  explicit HLTTauDQMPath(bool doRefAnalysis);
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
  const std::string& getLastFilterName() const { return std::get<0>(filterIndices_.back()); }
  // index (to edm::TriggerResults) of a filter
  size_t getFilterIndex(size_t i) const { return std::get<1>(filterIndices_[i]); }

  typedef std::tuple<std::string, size_t> FilterIndex;
private:
  const bool doRefAnalysis_;

  std::vector<boost::regex> pathRegexs_;
  std::vector<boost::regex> ignoreFilterTypes_;
  std::vector<boost::regex> ignoreFilterNames_;

  std::vector<FilterIndex> filterIndices_;
  std::string pathName_;
  unsigned int pathIndex_;
};


#endif
