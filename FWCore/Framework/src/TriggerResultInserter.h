#ifndef FrameworkTriggerResultsInserter_h
#define FrameworkTriggerResultsInserter_h

/*
  Author: Jim Kowalkowski 15-1-06

  This is an unusual module in that it is always present in the
  schedule and it is not configurable.
  The ownership of the bitmask is shared with the scheduler
  Its purpose is to create a TriggerResults instance and insert it into
  the event.

*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/TriggerResults.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "boost/shared_ptr.hpp"

#include <string>
#include <vector>

namespace edm
{
  class TriggerResultInserter : public edm::EDProducer
  {
  public:
    typedef std::vector<std::string> Strings;
    typedef boost::shared_ptr<TriggerResults::BitMask> BitMaskPtr;

    // standard constructor not supported for this module
    explicit TriggerResultInserter(edm::ParameterSet const& ps);

    // the pset needed here is the one that defines the trigger_path names
    // and the end_path names
    TriggerResultInserter(ParameterSet const& ps, BitMaskPtr);
    virtual ~TriggerResultInserter();

    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    BitMaskPtr bits_;
    // pset_id needed until run data exists
    ParameterSetID pset_id_;
    // pset_as_string needed until psets are stored in the output files
    Strings path_names_;
  };
}
#endif
