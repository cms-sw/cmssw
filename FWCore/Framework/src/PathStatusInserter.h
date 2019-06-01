#ifndef FWCore_Framework_PathStatusInserter_h
#define FWCore_Framework_PathStatusInserter_h

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include <vector>

namespace edm {

  class Event;
  class EventSetup;
  class StreamID;

  class PathStatusInserter : public global::EDProducer<> {
  public:
    PathStatusInserter(unsigned int numberOfStreams);

    void setPathStatus(StreamID const&, HLTPathStatus const&);

    void produce(StreamID, Event&, EventSetup const&) const final;

  private:
    std::vector<HLTPathStatus> hltPathStatus_;
    EDPutTokenT<HLTPathStatus> token_;
  };
}  // namespace edm
#endif
