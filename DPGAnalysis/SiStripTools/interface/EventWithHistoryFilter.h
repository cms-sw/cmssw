#ifndef DPGAnalysis_SiStripTools_EventWithHistoryFilter_H
#define DPGAnalysis_SiStripTools_EventWithHistoryFilter_H

#include <string>
#include <vector>
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class ParameterSet;
  class Event;
}
class EventWithHistory;

class EventWithHistoryFilter {

 public:
  EventWithHistoryFilter();
  EventWithHistoryFilter(const edm::ParameterSet& iConfig);

  void set(const edm::ParameterSet& iConfig);
  const bool selected(const EventWithHistory& he, const edm::EventSetup& iSetup) const;
  const bool selected(const EventWithHistory& he, const edm::Event& iEvent, const edm::EventSetup& iSetup) const;
  const bool selected(const edm::Event& event, const edm::EventSetup& iSetup) const;

 private:

  const bool is_selected(const EventWithHistory& he, const edm::EventSetup& iSetup, const std::vector<int>& apvphases) const;
  const int getAPVLatency( const edm::EventSetup& iSetup) const;
  const int getAPVMode( const edm::EventSetup& iSetup) const;
  const std::vector<int> getAPVPhase(const edm::Event& iEvent) const;
  const bool isAPVLatencyNotNeeded() const;
  const bool isAPVPhaseNotNeeded() const;
  const bool isAPVModeNotNeeded() const;
  const bool isCutInactive(const std::vector<int>& range) const;
  const bool isInRange(const long long bx, const std::vector<int>& range, const bool extra) const;
  void printConfig() const;

  edm::InputTag m_historyProduct;
  std::string m_partition;
  std::string m_APVPhaseLabel;
  std::vector<int> m_apvmodes;
  std::vector<int> m_dbxrange;
  std::vector<int> m_dbxrangelat;
  std::vector<int> m_bxrange;
  std::vector<int> m_bxrangelat;
  std::vector<int> m_bxcyclerange;
  std::vector<int> m_bxcyclerangelat;
  std::vector<int> m_dbxcyclerange;
  std::vector<int> m_dbxcyclerangelat;
  std::vector<int> m_dbxtrpltrange;
  std::vector<int> m_dbxgenericrange;
  unsigned int m_dbxgenericfirst;
  unsigned int m_dbxgenericlast;
  bool m_noAPVPhase;

};

#endif // DPGAnalysis_SiStripTools_EventWithHistoryFilter_H
