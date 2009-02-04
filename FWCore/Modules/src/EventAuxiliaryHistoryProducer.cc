
#include "FWCore/Modules/src/EventAuxiliaryHistoryProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm
{
  EventAuxiliaryHistoryProducer::EventAuxiliaryHistoryProducer(edm::ParameterSet const& ps):
    depth_(ps.getParameter<unsigned int>("historyDepth")),
    history_()
  {
      produces<std::vector<EventAuxiliary> > ();
  }

  EventAuxiliaryHistoryProducer::~EventAuxiliaryHistoryProducer()
  {
  }

  void EventAuxiliaryHistoryProducer::produce(edm::Event & e,edm::EventSetup const&)
  {
  EventAuxiliary aux( e.id(), "", e.time(), e.luminosityBlock(), e.isRealData(), e.experimentType(),
                      e.bunchCrossing(), EventAuxiliary::invalidStoreNumber,  e.orbitNumber()); 
  //const EventAuxiliary & aux = e.auxiliary(); // when available
  if(history_.size() > 0)
    {
    if(history_.back().id().next() != aux.id())    history_.clear();
    if(history_.size() >= depth_) history_.pop_front();
    }

   history_.push_back(aux);

   //Serialize into std::vector 
   std::vector<EventAuxiliary> *  out = new   std::vector<EventAuxiliary>;
   std::auto_ptr<std::vector<EventAuxiliary > > result(out);
   for(int j = 0 ; j < history_.size(); j++)  
   { 
    out->push_back(history_[j]);
   }
   e.put(result);

  }

  void EventAuxiliaryHistoryProducer::endJob()
  {
  }
}
