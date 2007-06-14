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

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "boost/shared_ptr.hpp"

#include <string>
#include <vector>

namespace edm
{
  class ParameterSet;
  class TriggerResultInserter : public edm::EDProducer
  {
  public:
    typedef std::vector<std::string> Strings;
    typedef std::vector<bool> Bools;
    typedef boost::shared_ptr<HLTGlobalStatus> TrigResPtr;

    // standard constructor not supported for this module
    explicit TriggerResultInserter(edm::ParameterSet const& ps);

    // the pset needed here is the one that defines the trigger_path names
    // and the end_path names
    TriggerResultInserter(edm::ParameterSet const& ps, const TrigResPtr& trptr);
    virtual ~TriggerResultInserter();

    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    TrigResPtr trptr_;
    // pset_id needed until run data exists
    ParameterSetID pset_id_;
    // pset_as_string needed until psets are stored in the output files
    Strings path_names_;
  };

  // ---------------------
#if 0
  // if and when the run information becomes avaialble from the event,
  // these function will need to be available and moved into an
  // algorithm or utility library that depends on the framework
  // The event is needed to get the run object
  // The trigger results are needed to get the trigger bits
  // These are free functions so they can exist in a separable library
  // that isolates the dependency on the Event class.
  // The initial implementation of these functions will just use the
  // information in the triggerresults object directly to answer the
  // questions.

  unsigned int   trigGetPosition(const Event& e, const TriggerResults& tr, const std::string& path_name);
  const std::string& trigGetName(const Event& e, const TriggerResults& tr, unsigned int bit_position);
  Bools              trigGetMask(const Event& e, const TriggerResults& tr, const Strings& path_names);

#endif


}
#endif
