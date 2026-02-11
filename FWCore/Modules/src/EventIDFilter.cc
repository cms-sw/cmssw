#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class EventIDFilter : public edm::global::EDFilter<> {
public:
  explicit EventIDFilter(edm::ParameterSet const& iConfig);

  bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const final;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::vector<edm::EventID> const ids_;
};

EventIDFilter::EventIDFilter(edm::ParameterSet const& iConfig)
    : ids_(iConfig.getParameter<std::vector<edm::EventID> >("eventsToPass")) {}

bool EventIDFilter::filter(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  edm::EventID const& id = iEvent.id();
  return std::find(ids_.begin(), ids_.end(), id) != ids_.end();
}

void EventIDFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::EventID> >("eventsToPass")->setComment("List of EventIDs to pass");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(EventIDFilter);