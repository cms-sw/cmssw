#ifndef FWCore_Framework_TriggerResultsInserter_h
#define FWCore_Framework_TriggerResultsInserter_h

/*
  Author: Jim Kowalkowski 15-1-06

  This is an unusual module in that it is always present in the
  schedule and it is not configurable.
  The ownership of the bitmask is shared with the scheduler
  Its purpose is to create a TriggerResults instance and insert it into
  the event.

*/

#include <vector>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include "boost/shared_ptr.hpp"

namespace edm
{
  class ParameterSet;
  class Event;
  class EventSetup;
  class HLTGlobalStatus;

  class TriggerResultInserter : public edm::global::EDProducer<>
  {
  public:

    typedef boost::shared_ptr<HLTGlobalStatus> TrigResPtr;

    // standard constructor not supported for this module
    explicit TriggerResultInserter(edm::ParameterSet const& ps);

    // the pset needed here is the one that defines the trigger path names
    TriggerResultInserter(edm::ParameterSet const& ps, unsigned int iNStreams);

    void setTrigResultForStream(unsigned int iStreamIndex,
                                const TrigResPtr& trptr);
    void produce(StreamID id, edm::Event& e, edm::EventSetup const& c) const override final;

  private:
    std::vector<TrigResPtr> resultsPerStream_;

    ParameterSetID pset_id_;
  };
}
#endif
