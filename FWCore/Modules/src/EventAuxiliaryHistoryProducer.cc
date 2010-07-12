#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <deque>

namespace edm {

  class EventAuxiliaryHistoryProducer : public EDProducer {
  public:
    explicit EventAuxiliaryHistoryProducer(ParameterSet const&);
    virtual ~EventAuxiliaryHistoryProducer();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    virtual void produce(Event& e, EventSetup const& c);
    void endJob();

  private:
    unsigned int depth_;
    std::deque<EventAuxiliary> history_; 
  };

  EventAuxiliaryHistoryProducer::EventAuxiliaryHistoryProducer(ParameterSet const& ps):
    depth_(ps.getParameter<unsigned int>("historyDepth")),
    history_() {
      produces<std::vector<EventAuxiliary> > ();
  }

  EventAuxiliaryHistoryProducer::~EventAuxiliaryHistoryProducer() {
  }

  void EventAuxiliaryHistoryProducer::produce(Event& e, EventSetup const&) {
    EventAuxiliary aux(e.id(), "", e.time(), e.isRealData(), e.experimentType(),
                       e.bunchCrossing(), EventAuxiliary::invalidStoreNumber, e.orbitNumber()); 
  //EventAuxiliary const& aux = e.auxiliary(); // when available
    if(history_.size() > 0) {
      if(history_.back().id().next(aux.luminosityBlock()) != aux.id()) history_.clear();
      if(history_.size() >= depth_) history_.pop_front();
    }

    history_.push_back(aux);

    //Serialize into std::vector 
    std::auto_ptr<std::vector<EventAuxiliary > > result(new std::vector<EventAuxiliary>);
    for(size_t j = 0; j < history_.size(); ++j) { 
      result->push_back(history_[j]);
    }
    e.put(result);
  }

  void EventAuxiliaryHistoryProducer::endJob() {
  }


  void
  EventAuxiliaryHistoryProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.add<unsigned int>("historyDepth");
    descriptions.add("EventAuxiliaryHistoryProducer", desc);
  }
}

using edm::EventAuxiliaryHistoryProducer;
DEFINE_FWK_MODULE(EventAuxiliaryHistoryProducer);
