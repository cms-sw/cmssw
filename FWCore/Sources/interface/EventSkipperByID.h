#ifndef FWCore_Sources_EventSkipperByID_h
#define FWCore_Sources_EventSkipperByID_h

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

#include <memory>
#include <vector>

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;

  class EventSkipperByID {
  public:
    explicit EventSkipperByID(ParameterSet const& pset);
    ~EventSkipperByID();
    bool skipIt(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const;
    bool skippingLumis() const {return skippingLumis_;}
    bool skippingEvents() const {return skippingEvents_;}
    bool somethingToSkip() const {return somethingToSkip_;}
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
    bool skippingLumis_;
    bool skippingEvents_;
    bool somethingToSkip_;
  };
}

#endif
