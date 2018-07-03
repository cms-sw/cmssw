#ifndef FWCore_Framework_EndPathStatusInserter_h
#define FWCore_Framework_EndPathStatusInserter_h

#include "FWCore/Framework/interface/global/EDProducer.h"

namespace edm {

  class Event;
  class EventSetup;
  class StreamID;

  class EndPathStatusInserter : public global::EDProducer<> {
  public:

    EndPathStatusInserter(unsigned int numberOfStreams);

    void produce(StreamID, Event&, EventSetup const&) const final;
  };
}
#endif
