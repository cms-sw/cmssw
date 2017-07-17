#ifndef FWCore_Framework_PathStatusInserter_h
#define FWCore_Framework_PathStatusInserter_h

#include <vector>
#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

namespace edm {

class Event;
class EventSetup;
class StreamID;

class PathStatusInserter : public global::EDProducer<> {
 public:
  PathStatusInserter(unsigned int numberOfStreams);

  void setPathStatus(StreamID const&, HLTPathStatus const&);

  void produce(StreamID, Event&, EventSetup const&) const override final;

 private:
  std::vector<HLTPathStatus> hltPathStatus_;
};
}
#endif
