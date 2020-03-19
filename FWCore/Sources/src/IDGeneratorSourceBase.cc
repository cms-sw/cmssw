/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <cerrno>

#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ExceptionHelpers.h"
#include "FWCore/Sources/interface/IDGeneratorSourceBase.h"

#include "FWCore/Sources/interface/PuttableSourceBase.h"

namespace {
  void checkFirstLumiForRuns(std::vector<edm::LuminosityBlockID> const& iFirstLumis) {
    if (iFirstLumis.empty())
      return;

    auto previous = iFirstLumis[0].luminosityBlock();
    for (auto it = iFirstLumis.begin() + 1; it != iFirstLumis.end(); ++it) {
      if (not(it->luminosityBlock() > previous)) {
        throw edm::Exception(edm::errors::Configuration)
            << "Incorrect ordering of LuminosityBlock numbers in parameter 'firstLuminosityBlockForEachRun'";
      }
    }
  }
}  // namespace
namespace edm {
  //used for defaults
  static unsigned long long constexpr kNanoSecPerSec = 1000000000ULL;
  static unsigned long long constexpr kAveEventPerSec = 200ULL;

  template <typename BASE>
  IDGeneratorSourceBase<BASE>::IDGeneratorSourceBase(ParameterSet const& pset,
                                                     InputSourceDescription const& desc,
                                                     bool realData)
      : BASE(pset, desc),
        firstLumiForRuns_(
            pset.getUntrackedParameter<std::vector<edm::LuminosityBlockID>>("firstLuminosityBlockForEachRun")),
        numberEventsInRun_(pset.getUntrackedParameter<unsigned int>("numberEventsInRun", BASE::remainingEvents())),
        numberEventsInLumi_(
            pset.getUntrackedParameter<unsigned int>("numberEventsInLuminosityBlock", BASE::remainingEvents())),
        presentTime_(pset.getUntrackedParameter<unsigned long long>("firstTime", 1ULL)),  //time in ns
        origTime_(presentTime_),
        timeBetweenEvents_(
            pset.getUntrackedParameter<unsigned long long>("timeBetweenEvents", kNanoSecPerSec / kAveEventPerSec)),
        eventCreationDelay_(pset.getUntrackedParameter<unsigned int>("eventCreationDelay", 0)),
        numberEventsInThisRun_(0),
        numberEventsInThisLumi_(0),
        zerothEvent_(pset.existsAs<unsigned int>("firstEvent", false)
                         ? pset.getUntrackedParameter<unsigned int>("firstEvent", 1) - 1
                         : pset.getUntrackedParameter<unsigned long long>("firstEvent", 1) - 1),
        eventID_(pset.getUntrackedParameter<unsigned int>("firstRun", 1),
                 pset.getUntrackedParameter<unsigned int>("firstLuminosityBlock", 1),
                 zerothEvent_),
        origEventID_(eventID_),
        isRealData_(realData),
        eType_(EventAuxiliary::Undefined) {
    BASE::setTimestamp(Timestamp(presentTime_));
    checkFirstLumiForRuns(firstLumiForRuns_);
    if (not firstLumiForRuns_.empty()) {
      numberEventsInRun_ = -1;
      eventID_ = EventID(runForLumi(eventID_.luminosityBlock()), eventID_.luminosityBlock(), zerothEvent_);
    }
  }

  template <typename BASE>
  IDGeneratorSourceBase<BASE>::~IDGeneratorSourceBase() noexcept(false) {}

  template <typename BASE>
  std::shared_ptr<RunAuxiliary> IDGeneratorSourceBase<BASE>::readRunAuxiliary_() {
    Timestamp ts = Timestamp(presentTime_);
    BASE::resetNewRun();
    return std::make_shared<RunAuxiliary>(eventID_.run(), ts, Timestamp::invalidTimestamp());
  }

  template <typename BASE>
  std::shared_ptr<LuminosityBlockAuxiliary> IDGeneratorSourceBase<BASE>::readLuminosityBlockAuxiliary_() {
    if (BASE::processingMode() == BASE::Runs)
      return std::shared_ptr<LuminosityBlockAuxiliary>();
    Timestamp ts = Timestamp(presentTime_);
    BASE::resetNewLumi();
    return std::make_shared<LuminosityBlockAuxiliary>(
        eventID_.run(), eventID_.luminosityBlock(), ts, Timestamp::invalidTimestamp());
  }

  template <typename BASE>
  void IDGeneratorSourceBase<BASE>::skip(int offset) {
    EventID oldEventID = eventID_;
    for (; offset < 0; ++offset) {
      retreatToPrevious(eventID_, presentTime_);
    }
    for (; offset > 0; --offset) {
      advanceToNext(eventID_, presentTime_);
    }
    if (eventID_.run() != oldEventID.run()) {
      // New Run
      BASE::setNewRun();
      BASE::setNewLumi();
    }
    if (eventID_.luminosityBlock() != oldEventID.luminosityBlock()) {
      // New Lumi
      BASE::setNewLumi();
    }
  }

  template <typename BASE>
  void IDGeneratorSourceBase<BASE>::beginJob() {
    BASE::beginJob();
    // Initialize cannot be called from the constructor, because it is a virtual function
    // that needs to be invoked from a derived class if the derived class overrides it.
    initialize(eventID_, presentTime_, timeBetweenEvents_);
  }

  template <typename BASE>
  void IDGeneratorSourceBase<BASE>::initialize(EventID&, TimeValue_t&, TimeValue_t&) {}

  template <typename BASE>
  void IDGeneratorSourceBase<BASE>::rewind_() {
    presentTime_ = origTime_;
    eventID_ = origEventID_;
    numberEventsInThisRun_ = 0;
    numberEventsInThisLumi_ = 0;
    BASE::setNewRun();
    BASE::setNewLumi();
  }

  template <typename BASE>
  typename BASE::ItemType IDGeneratorSourceBase<BASE>::getNextItemType() {
    if (BASE::state() == BASE::IsInvalid) {
      return noFiles() ? BASE::IsStop : BASE::IsFile;
    }
    if (BASE::newRun()) {
      return BASE::IsRun;
    }
    if (BASE::newLumi()) {
      return BASE::IsLumi;
    }
    if (BASE::eventCached()) {
      return BASE::IsEvent;
    }
    EventID oldEventID = eventID_;
    advanceToNext(eventID_, presentTime_);
    if (eventCreationDelay_ > 0) {
      usleep(eventCreationDelay_);
    }
    size_t index = fileIndex();
    bool another = setRunAndEventInfo(eventID_, presentTime_, eType_);
    if (!another) {
      return BASE::IsStop;
    }
    bool newFile = (fileIndex() > index);
    BASE::setEventCached();
    if (BASE::newRun() || eventID_.run() != oldEventID.run()) {
      // New Run
      BASE::setNewRun();
      BASE::setNewLumi();
      return newFile ? BASE::IsFile : BASE::IsRun;
    }
    if (BASE::processingMode() == BASE::Runs) {
      return newFile ? BASE::IsFile : BASE::IsRun;
    }
    if (BASE::processingMode() == BASE::RunsAndLumis) {
      return newFile ? BASE::IsFile : BASE::IsLumi;
    }
    // Same Run
    if (BASE::newLumi() || eventID_.luminosityBlock() != oldEventID.luminosityBlock()) {
      // New Lumi
      BASE::setNewLumi();
      return newFile ? BASE::IsFile : BASE::IsLumi;
    }
    return newFile ? BASE::IsFile : BASE::IsEvent;
  }

  template <typename BASE>
  void IDGeneratorSourceBase<BASE>::advanceToNext(EventID& eventID, TimeValue_t& time) {
    if (numberEventsInRun_ < 1 || numberEventsInThisRun_ < numberEventsInRun_) {
      // same run
      ++numberEventsInThisRun_;
      if (!(numberEventsInLumi_ < 1 || numberEventsInThisLumi_ < numberEventsInLumi_)) {
        // new lumi
        eventID = eventID.next(eventID.luminosityBlock() + 1);
        numberEventsInThisLumi_ = 1;
        if (not firstLumiForRuns_.empty()) {
          auto run = runForLumi(eventID.luminosityBlock());
          if (run != eventID.run()) {
            numberEventsInThisRun_ = 1;
            eventID = eventID.nextRunFirstEvent(eventID_.luminosityBlock());

            eventID = EventID(run, eventID.luminosityBlock(), eventID.event());
          }
        }
      } else {
        eventID = eventID.next(eventID.luminosityBlock());
        ++numberEventsInThisLumi_;
      }
    } else {
      // new run
      eventID = eventID.nextRunFirstEvent(origEventID_.luminosityBlock());
      numberEventsInThisLumi_ = 1;
      numberEventsInThisRun_ = 1;
    }
    time += timeBetweenEvents_;
  }

  template <typename BASE>
  void IDGeneratorSourceBase<BASE>::retreatToPrevious(EventID& eventID, TimeValue_t& time) {
    if (numberEventsInRun_ < 1 || numberEventsInThisRun_ > 0) {
      // same run
      --numberEventsInThisRun_;
      eventID = eventID.previous(eventID.luminosityBlock());
      if (!(numberEventsInLumi_ < 1 || numberEventsInThisLumi_ > 0)) {
        // new lumi
        eventID = eventID.previous(eventID.luminosityBlock() - 1);
        numberEventsInThisLumi_ = numberEventsInLumi_;

        if (not firstLumiForRuns_.empty()) {
          auto run = runForLumi(eventID.luminosityBlock());
          if (run != eventID.run()) {
            eventID = eventID.previousRunLastEvent(eventID_.luminosityBlock());

            eventID = EventID(run, eventID.luminosityBlock(), eventID.event());
          }
        }
      } else {
        --numberEventsInThisLumi_;
      }
    } else {
      // new run
      assert(numberEventsInLumi_ != 0);
      eventID = eventID.previousRunLastEvent(origEventID_.luminosityBlock() + numberEventsInRun_ / numberEventsInLumi_);
      eventID = EventID(numberEventsInRun_, eventID.luminosityBlock(), eventID.run());
      numberEventsInThisLumi_ = numberEventsInLumi_;
      numberEventsInThisRun_ = numberEventsInRun_;
    }
    time -= timeBetweenEvents_;
  }

  template <typename BASE>
  RunNumber_t IDGeneratorSourceBase<BASE>::runForLumi(LuminosityBlockNumber_t iLumi) const {
    auto it = std::find_if(firstLumiForRuns_.rbegin(), firstLumiForRuns_.rend(), [iLumi](auto const& iV) {
      return iV.luminosityBlock() <= iLumi;
    });
    if (it == firstLumiForRuns_.rend()) {
      //use first since we are off the end
      return firstLumiForRuns_[0].run();
    }
    return it->run();
  }

  template <typename BASE>
  bool IDGeneratorSourceBase<BASE>::noFiles() const {
    return false;
  }

  template <typename BASE>
  size_t IDGeneratorSourceBase<BASE>::fileIndex() const {
    return 0UL;
  }

  template <typename BASE>
  void IDGeneratorSourceBase<BASE>::fillDescription(ParameterSetDescription& desc) {
    desc.addOptionalUntracked<unsigned int>("numberEventsInRun")
        ->setComment("Number of events to generate in each run.");
    desc.addOptionalUntracked<unsigned int>("numberEventsInLuminosityBlock")
        ->setComment("Number of events to generate in each lumi.");
    desc.addUntracked<unsigned long long>("firstTime", 1)->setComment("Time before first event (ns) (for timestamp).");
    desc.addUntracked<unsigned long long>("timeBetweenEvents", kNanoSecPerSec / kAveEventPerSec)
        ->setComment("Time between consecutive events (ns) (for timestamp).");
    desc.addUntracked<unsigned int>("eventCreationDelay", 0)
        ->setComment("Real time delay between generation of consecutive events (ms).");

    desc.addNode(edm::ParameterDescription<unsigned int>("firstEvent", 1U, false) xor
                 edm::ParameterDescription<unsigned long long>("firstEvent", 1ULL, false))
        ->setComment(
            "'firstEvent' is an XOR group because it can have type uint32 or uint64, default:1\n"
            "Event number of first event to generate.");

    desc.addUntracked<unsigned int>("firstLuminosityBlock", 1)
        ->setComment("Luminosity block number of first lumi to generate.");
    desc.addUntracked<unsigned int>("firstRun", 1)->setComment("Run number of first run to generate.");
    desc.addUntracked<std::vector<edm::LuminosityBlockID>>("firstLuminosityBlockForEachRun", {})
        ->setComment(
            "When the source makes a new LuminosityBlock, this list is checked to see what Run number should be used. "
            "The LuminosityBlock numbers are required to be in ascending order.");
    BASE::fillDescription(desc);
  }

  template class IDGeneratorSourceBase<PuttableSourceBase>;
  template class IDGeneratorSourceBase<InputSource>;
}  // namespace edm
