#ifndef DQM_TrackingMonitorSource_HLTPathSelector_h
#define DQM_TrackingMonitorSource_HLTPathSelector_h

#include <vector>
#include <string>
#include <map>
#include "TPRegexp.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

class HLTPathSelector : public edm::stream::EDFilter<> {
public:
  explicit HLTPathSelector(const edm::ParameterSet&);

private:
  void beginRun(edm::Run const &, edm::EventSetup const&) override;
  bool filter(edm::Event&, edm::EventSetup const&) override;
  void endJob() ;

private:

  // module config parameters
  bool verbose_;
  std::string processName_;
  const std::vector<std::string> hltPathsOfInterest_;
  edm::InputTag triggerResultsTag_;
  edm::InputTag triggerEventTag_;
  const edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  const edm::EDGetTokenT<trigger::TriggerEvent> triggerEventToken_;

  HLTConfigProvider hltConfig_;

  std::map<std::string, unsigned int> hltPathsMap_;
  std::map<std::string, int> tmap_;
};
#endif
