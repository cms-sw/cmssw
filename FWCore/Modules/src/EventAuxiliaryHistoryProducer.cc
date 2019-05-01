#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include <deque>

namespace edm {

  class EventAuxiliaryHistoryProducer : public one::EDProducer<> {
  public:
    explicit EventAuxiliaryHistoryProducer(ParameterSet const&);

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    void produce(Event& e, EventSetup const& c) override;
    void endJob() override;

  private:
    unsigned int depth_;
    std::deque<EventAuxiliary> history_;
    EDPutTokenT<std::vector<EventAuxiliary>> token_;
  };

  EventAuxiliaryHistoryProducer::EventAuxiliaryHistoryProducer(ParameterSet const& ps)
      : depth_(ps.getParameter<unsigned int>("historyDepth")),
        history_(),
        token_{produces<std::vector<EventAuxiliary>>()} {}

  void EventAuxiliaryHistoryProducer::produce(Event& e, EventSetup const&) {
    EventAuxiliary aux(e.id(),
                       "",
                       e.time(),
                       e.isRealData(),
                       e.experimentType(),
                       e.bunchCrossing(),
                       EventAuxiliary::invalidStoreNumber,
                       e.orbitNumber());
    //EventAuxiliary const& aux = e.auxiliary(); // when available
    if (!history_.empty()) {
      if (history_.back().id().next(aux.luminosityBlock()) != aux.id())
        history_.clear();
      if (history_.size() >= depth_)
        history_.pop_front();
    }

    history_.push_back(aux);

    e.emplace(token_, history_.begin(), history_.end());
  }

  void EventAuxiliaryHistoryProducer::endJob() {}

  void EventAuxiliaryHistoryProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.add<unsigned int>("historyDepth");
    descriptions.add("eventAuxiliaryHistory", desc);
  }
}  // namespace edm

using edm::EventAuxiliaryHistoryProducer;
DEFINE_FWK_MODULE(EventAuxiliaryHistoryProducer);
