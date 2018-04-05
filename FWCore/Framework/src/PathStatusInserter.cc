
#include "FWCore/Framework/src/PathStatusInserter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <memory>

namespace edm
{
  PathStatusInserter::PathStatusInserter(unsigned int numberOfStreams) :
    hltPathStatus_(numberOfStreams)
  {
    produces<HLTPathStatus>();
  }

  void
  PathStatusInserter::setPathStatus(StreamID const& streamID,
                                    HLTPathStatus const& hltPathStatus) {
    hltPathStatus_[streamID.value()] = hltPathStatus;
  }

  void
  PathStatusInserter::produce(StreamID streamID, edm::Event& event, edm::EventSetup const&) const
  {
    event.put(std::make_unique<HLTPathStatus>(hltPathStatus_[streamID.value()]));
  }
}
