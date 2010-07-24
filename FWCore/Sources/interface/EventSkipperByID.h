#ifndef FWCore_Sources_EventSkipperByID_h
#define FWCore_Sources_EventSkipperByID_h

#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "boost/shared_ptr.hpp"
#include <memory>
#include <vector>

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;

  class EventSkipperByID {
  public:
    explicit EventSkipperByID(ParameterSet const& pset);
    ~EventSkipperByID();
    bool operator()(LuminosityBlockRange const& lumiRange) const;
    bool operator()(EventRange const& eventRange) const;
    bool skipIt(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const;
    bool somethingToSkip() const;
    bool skippingLumis() const;
    static
    std::auto_ptr<EventSkipperByID>create(ParameterSet const& pset);
    static void fillDescription(ParameterSetDescription & desc);

  private:

    RunNumber_t firstRun_;
    LuminosityBlockNumber_t firstLumi_;
    EventNumber_t firstEvent_;
    std::vector<LuminosityBlockRange> whichLumisToSkip_;
    std::vector<LuminosityBlockRange> whichLumisToProcess_;
    std::vector<EventRange> whichEventsToSkip_;
    std::vector<EventRange> whichEventsToProcess_;
    mutable LuminosityBlockID lumi_;
    mutable MinimalEventID event_;
  };
}

#endif
