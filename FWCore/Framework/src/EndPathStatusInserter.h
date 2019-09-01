#ifndef FWCore_Framework_EndPathStatusInserter_h
#define FWCore_Framework_EndPathStatusInserter_h

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

namespace edm {

  class Event;
  class EventSetup;
  class StreamID;
  class EndPathStatus;

  class EndPathStatusInserter : public global::EDProducer<> {
  public:
    explicit EndPathStatusInserter(unsigned int numberOfStreams);

    void produce(StreamID, Event&, EventSetup const&) const final;

  private:
    EDPutTokenT<EndPathStatus> token_;
  };
}  // namespace edm
#endif
