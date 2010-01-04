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
	lumi_(),
	event_() {
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

  bool
  EventSkipperByID::somethingToSkip() const {
    return !(firstRun_ <= 1U && firstLumi_ <= 1U && firstEvent_ <= 1U &&
      whichLumisToSkip_.empty() && whichLumisToProcess_.empty() && whichEventsToSkip_.empty() && whichEventsToProcess_.empty());
  }

  bool
  EventSkipperByID::operator()(LuminosityBlockRange const& lumiRange) const {
    return contains(lumiRange, lumi_);
  }
  bool
  EventSkipperByID::operator()(EventRange const& eventRange) const {
    return contains(eventRange, event_);
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
    // If we get here, the entry was not skipped due to firstRun, firstLuminosityBlock, and/or firstEvent.
    lumi_ = LuminosityBlockID(run, lumi);
    if(search_if_in_all(whichLumisToSkip_, *this)) {
      // The entry is in a lumi specified in whichLumisToSkip.  Skip it.
      return true;
    }
    if(!whichLumisToProcess_.empty() && !search_if_in_all(whichLumisToProcess_, *this)) {
      // The entry is not in a lumi specified in non-empty whichLumisToProcess.  Skip it.
      return true;
    }
    if(event == 0U) {
      // The entry is a lumi entry that was not skipped above.  Keep it.
      return false;
    }
    event_ = MinimalEventID(run, event);
    if(search_if_in_all(whichEventsToSkip_, *this)) {
      // The event is specified in whichEventsToSkip.  Skip it.
      return true;
    }
    if(!whichEventsToProcess_.empty() && !search_if_in_all(whichEventsToProcess_, *this)) {
      // The event is not specified in non-empty whichEventsToProcess.  Skip it.
      return true;
    }
    return false;
  }

  bool
  EventSkipperByID::operator()(FileIndex::Element& e) const {
    return skipIt(e.run_, e.lumi_, e.event_);
  }

  void
  EventSkipperByID::fillDescription(ParameterSetDescription & desc) {

    desc.addUntracked<unsigned int>("firstRun", 1U);
    desc.addUntracked<unsigned int>("firstLuminosityBlock", 0U);
    desc.addUntracked<unsigned int>("firstEvent", 1U);

    std::vector<LuminosityBlockRange> defaultLumis;
    desc.addUntracked<std::vector<LuminosityBlockRange> >("lumisToSkip", defaultLumis);
    desc.addUntracked<std::vector<LuminosityBlockRange> >("lumisToProcess", defaultLumis);

    std::vector<EventRange> defaultEvents;
    desc.addUntracked<std::vector<EventRange> >("eventsToSkip", defaultEvents);
    desc.addUntracked<std::vector<EventRange> >("eventsToProcess", defaultEvents);
  }
}
