#include "FWCore/Sources/interface/EventSkipperByID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  EventSkipperByID::EventSkipperByID(ParameterSet const& pset) :
	// The default value provided as the second argument to the getUntrackedParameter function call
	// is not used when the ParameterSet has been validated and the parameters are not optional
	// in the description.  As soon as all primary input sources and all modules with a secondary
	// input sources have defined descriptions, the defaults in the getUntrackedParameterSet function
	// calls can and should be deleted from the code.
        firstRun_(pset.getUntrackedParameter<unsigned int>("firstRun", 1U)),
        firstLumi_(pset.getUntrackedParameter<unsigned int>("firstLuminosityBlock", 0U)),
        firstEvent_(pset.getUntrackedParameter<unsigned int>("firstEvent", 1U)),
        whichLumisToSkip_(pset.getUntrackedParameter<std::vector<LuminosityBlockRange> >("lumisToSkip", std::vector<LuminosityBlockRange>())),
        whichLumisToProcess_(pset.getUntrackedParameter<std::vector<LuminosityBlockRange> >("lumisToProcess", std::vector<LuminosityBlockRange>())),
        whichEventsToSkip_(pset.getUntrackedParameter<std::vector<EventRange> >("eventsToSkip",std::vector<EventRange>())),
        whichEventsToProcess_(pset.getUntrackedParameter<std::vector<EventRange> >("eventsToProcess",std::vector<EventRange>())),
        skippingLumis_(!(whichLumisToSkip_.empty() && whichLumisToProcess_.empty())),
        skippingEvents_(!(whichEventsToSkip_.empty() && whichEventsToProcess_.empty())),
        somethingToSkip_(skippingLumis_ || skippingEvents_ || !(firstRun_ <= 1U && firstLumi_ <= 1U && firstEvent_ <= 1U)) {
    sortAndRemoveOverlaps(whichLumisToSkip_);
    sortAndRemoveOverlaps(whichLumisToProcess_);
    sortAndRemoveOverlaps(whichEventsToSkip_);
    sortAndRemoveOverlaps(whichEventsToProcess_);
  }

  EventSkipperByID::~EventSkipperByID() {}

  std::auto_ptr<EventSkipperByID>
  EventSkipperByID::create(ParameterSet const& pset) {
    std::auto_ptr<EventSkipperByID> evSkp(new EventSkipperByID(pset));
    if (!evSkp->somethingToSkip()) {
      evSkp.reset();
    }
    return evSkp;
  }

  // Determines whether a run, lumi, or event will be skipped based on the run, lumi, and event number.
  // This function is called by a predicate, so it must not modify the state of the EventSkipperByID_ object.
  // The mutable lumi_ and event_ data members are just temporary caches, so they may be modified.
  bool
  EventSkipperByID::skipIt(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const {

    if(run == 0U) run = 1U; // Correct zero run number
    if(run < firstRun_) {
      // Skip all entries before the first run.
      return true;
    }
    if(lumi == 0U) {
      // This is a run entry that is not before the first run.
      // Keep it, since there are no other parameters to skip runs.
      return false;
    }
    // If we get here, this is a lumi or event entry.
    if(run == firstRun_) {
      // This lumi or event entry is in the first run to be processed.
      if(lumi < firstLumi_) {
        // This lumi or event entry is for a lumi prior to the first lumi to be processed.  Skip it.
        return true;
      }
      if(firstLumi_ == 0 || lumi == firstLumi_) {
        // If we get here, this entry is in the first lumi to be processed in the first run.
	// Note that if firstLumi_ == 0, we are processing all lumis in the run.
        if(event != 0U && event < firstEvent_) {
	  // This is an event entry prior to the first event to be processed. Skip it.
          return true;
        }
      }
    }
    if (skippingLumis()) {
      // If we get here, the entry was not skipped due to firstRun, firstLuminosityBlock, and/or firstEvent.
      LuminosityBlockID lumiID = LuminosityBlockID(run, lumi);
      LuminosityBlockRange lumiRange = LuminosityBlockRange(lumiID, lumiID);
      bool(*lt)(LuminosityBlockRange const&, LuminosityBlockRange const&) = &lessThan;
      if(binary_search_all(whichLumisToSkip_, lumiRange, lt)) {
        // The entry is in a lumi specified in whichLumisToSkip.  Skip it.
        return true;
      }
      if(!whichLumisToProcess_.empty() && !binary_search_all(whichLumisToProcess_, lumiRange, lt)) {
        // The entry is not in a lumi specified in non-empty whichLumisToProcess.  Skip it.
        return true;
      }
    }
    if(event == 0U) {
      // The entry is a lumi entry that was not skipped above.  Keep it.
      return false;
    }
    if (skippingEvents()) {
      // If we get here, the entry was not skipped due to firstRun, firstLuminosityBlock, and/or firstEvent.
      EventID eventID = EventID(run, lumi, event);
      EventRange eventRange = EventRange(eventID, eventID);
      EventID eventIDNoLumi = EventID(run, 0U, event);
      EventRange eventRangeNoLumi = EventRange(eventIDNoLumi, eventIDNoLumi);
      bool(*lt)(EventRange const&, EventRange const&) = &lessThanSpecial;
      if(binary_search_all(whichEventsToSkip_, eventRange, lt) || binary_search_all(whichEventsToSkip_, eventRangeNoLumi, lt)) {
        // The entry is an event specified in whichEventsToSkip.  Skip it.
        return true;
      }
      if(!whichEventsToProcess_.empty() && !binary_search_all(whichEventsToProcess_, eventRange, lt) && !binary_search_all(whichEventsToProcess_, eventRangeNoLumi, lt)) {
        // The entry is not an event specified in non-empty whichEventsToProcess.  Skip it.
        return true;
      }
    }
    return false;
  }

  void
  EventSkipperByID::fillDescription(ParameterSetDescription & desc) {

    desc.addUntracked<unsigned int>("firstRun", 1U)
        ->setComment("Skip any run with run number < 'firstRun'.");
    desc.addUntracked<unsigned int>("firstLuminosityBlock", 0U)
        ->setComment("Skip any lumi in run 'firstRun' with lumi number < 'firstLuminosityBlock'.");
    desc.addUntracked<unsigned int>("firstEvent", 1U)
        ->setComment("If 'firstLuminosityBlock' == 0, skip any event in run 'firstRun' with event number < 'firstEvent'.\n"
                     "If 'firstLuminosityBlock' != 0, skip any event in lumi 'firstRun:firstLuminosityBlock' with event number < 'firstEvent'.");

    std::vector<LuminosityBlockRange> defaultLumis;
    desc.addUntracked<std::vector<LuminosityBlockRange> >("lumisToSkip", defaultLumis)
        ->setComment("Skip any lumi inside the specified run:lumi range. In python do 'help(cms.LuminosityBlockRange)' for documentation.");
    desc.addUntracked<std::vector<LuminosityBlockRange> >("lumisToProcess", defaultLumis)
        ->setComment("If not empty, skip any lumi outside the specified run:lumi range. In python do 'help(cms.LuminosityBlockRange)' for documentation.");

    std::vector<EventRange> defaultEvents;
    desc.addUntracked<std::vector<EventRange> >("eventsToSkip", defaultEvents)
        ->setComment("Skip any event inside the specified run:event or run:lumi:event range. In python do 'help(cms.EventRange)' for documentation.");
    desc.addUntracked<std::vector<EventRange> >("eventsToProcess", defaultEvents)
        ->setComment("If not empty, skip any event outside the specified run:event or run:lumi:event range. In python do 'help(cms.EventRange)' for documentation.");
  }
}
