
#include "FWCore/Framework/src/PathStatusInserter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <memory>

namespace edm
{
  PathStatusInserter::PathStatusInserter(unsigned int numberOfStreams) :
    hltPathStatus_(numberOfStreams),
    token_{produces<HLTPathStatus>()}
  {
  }

  void
  PathStatusInserter::setPathStatus(StreamID const& streamID,
                                    HLTPathStatus const& hltPathStatus) {
    hltPathStatus_[streamID.value()] = hltPathStatus;
  }

  void
  PathStatusInserter::produce(StreamID streamID, edm::Event& event, edm::EventSetup const&) const
  {
    event.emplace(token_,hltPathStatus_[streamID.value()]);
  }
}
