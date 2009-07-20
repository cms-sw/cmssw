#ifndef DPGAnalysis_SiStripTools_EventWithHistoryFilter_H
#define DPGAnalysis_SiStripTools_EventWithHistoryFilter_H

#include <string>
#include <vector>
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class ParameterSet;
  class Event;
  class Run;
}
class EventWithHistory;

class EventWithHistoryFilter {

 public:
  EventWithHistoryFilter();
  EventWithHistoryFilter(const edm::ParameterSet& iConfig);

  void set(const edm::ParameterSet& iConfig);
  const bool selected(const EventWithHistory& he, const edm::EventSetup& iSetup) const;
  const bool selected(const EventWithHistory& he, const edm::Run& iRun, const edm::EventSetup& iSetup) const;
  const bool selected(const edm::Event& event, const edm::EventSetup& iSetup) const;

 private:

  const bool is_selected(const EventWithHistory& he, const edm::EventSetup& iSetup, const int apvphase) const;
  const int getAPVLatency( const edm::EventSetup& iSetup) const;
  const int getAPVPhase(const edm::Run& iRun) const;
  const bool isAPVLatencyNotNeeded() const;
  const bool isAPVPhaseNotNeeded() const;
  const bool isCutInactive(const std::vector<int>& range) const;
  const bool isInRange(const long long bx, const std::vector<int>& range, const bool extra) const;
  void printConfig() const;

  edm::InputTag _historyProduct;
  std::string _partition;
  std::vector<int> _dbxrange;
  std::vector<int> _dbxrangelat;
  std::vector<int> _bxrange;
  std::vector<int> _bxrangelat;
  std::vector<int> _bxcyclerange;
  std::vector<int> _bxcyclerangelat;
  std::vector<int> _dbxcyclerange;
  std::vector<int> _dbxcyclerangelat;
  std::vector<int> _dbxtrpltrange;
  bool _noAPVPhase;

};

#endif // DPGAnalysis_SiStripTools_EventWithHistoryFilter_H
