#ifndef ModulesEventAuxiliaryHistoryProducer_H
#define ModulesEventAuxiliaryHistoryProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include <deque>

namespace edm
{
  class EventAuxiliaryHistoryProducer : public edm::EDProducer
  {
  public:
    explicit EventAuxiliaryHistoryProducer(edm::ParameterSet const&);
    virtual ~EventAuxiliaryHistoryProducer();

    virtual void produce(edm::Event& e, edm::EventSetup const& c);
    void endJob();

  private:
    unsigned int depth_;
    std::deque<edm::EventAuxiliary> history_; 
  };
}

#endif

