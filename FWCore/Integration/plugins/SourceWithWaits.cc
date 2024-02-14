// -*- C++ -*-
//
// Package:     FWCore/Integration
// Class  :     SourceWithWaits
//
// Original Author:  W. David Dagenhart
//         Created:  12 October 2023

// This source allows configuring both a time per lumi section
// and events per lumi. Calls to usleep are inserted in the
// getNextItemType function in the amount
//
//   (time per lumi) / (events per lumi + 1)
//
// The sleeps occur before getNextItemType returns when
// an event is next and also when a lumi is next (excluding
// the first lumi). The total time sleeping that elapses per
// lumi is approximately equal to the configured amount.
// The algorithm accomplishing this is not perfect and
// if the events take enough time to process, then the lumis
// will last longer than configured amount (just because
// that was a lot easier to implement and good enough for
// the test this is used for).
//
// The time per lumi is the same for all lumis. events per lumi
// can be different each lumi. You can also configure a single
// value for lumis per run if you want multiple runs.
//
// The job will stop when the end of the vector specifying
// events per lumi is reached (it might end earlier if maxEvents
// is also configured).
//
// In some ways this source is like EmptySource. It does not produce
// or read anything. The initial intent is to use for tests of
// some issues we are facing with concurrent lumis in the online
// source. It emulates the relevant behavior of that source without
// all the associated complexity.

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <cassert>
#include <memory>
#include <unistd.h>
#include <vector>

namespace edmtest {
  class SourceWithWaits : public edm::InputSource {
  public:
    explicit SourceWithWaits(edm::ParameterSet const&, edm::InputSourceDescription const&);
    ~SourceWithWaits() override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    edm::InputSource::ItemTypeInfo getNextItemType() override;
    std::shared_ptr<edm::RunAuxiliary> readRunAuxiliary_() override;
    std::shared_ptr<edm::LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;
    void readEvent_(edm::EventPrincipal&) override;

    double timePerLumi_;           // seconds
    double sleepAfterStartOfRun_;  // seconds
    std::vector<unsigned int> eventsPerLumi_;
    unsigned int lumisPerRun_;
    unsigned int multipleEntriesForRun_;
    unsigned int multipleEntriesForLumi_;
    bool declareLast_;
    bool declareAllLast_;

    edm::EventNumber_t currentEvent_ = 0;
    edm::LuminosityBlockNumber_t currentLumi_ = 0;
    edm::RunNumber_t currentRun_ = 0;
    unsigned int currentFile_ = 0;
    unsigned int eventInCurrentLumi_ = 0;
    unsigned int lumiInCurrentRun_ = 0;
    bool startedNewRun_ = false;
    bool lastEventOfLumi_ = false;
    bool noEventsInLumi_ = false;
  };

  SourceWithWaits::SourceWithWaits(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc)
      : edm::InputSource(pset, desc),
        timePerLumi_(pset.getUntrackedParameter<double>("timePerLumi")),
        sleepAfterStartOfRun_(pset.getUntrackedParameter<double>("sleepAfterStartOfRun")),
        eventsPerLumi_(pset.getUntrackedParameter<std::vector<unsigned int>>("eventsPerLumi")),
        lumisPerRun_(pset.getUntrackedParameter<unsigned int>("lumisPerRun")),
        multipleEntriesForRun_(pset.getUntrackedParameter<unsigned int>("multipleEntriesForRun")),
        multipleEntriesForLumi_(pset.getUntrackedParameter<unsigned int>("multipleEntriesForLumi")),
        declareLast_(pset.getUntrackedParameter<bool>("declareLast")),
        declareAllLast_(pset.getUntrackedParameter<bool>("declareAllLast")) {}

  SourceWithWaits::~SourceWithWaits() {}

  void SourceWithWaits::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<double>("timePerLumi");
    desc.addUntracked<double>("sleepAfterStartOfRun");
    desc.addUntracked<std::vector<unsigned int>>("eventsPerLumi");
    desc.addUntracked<unsigned int>("lumisPerRun");
    desc.addUntracked<unsigned int>("multipleEntriesForRun", 0);
    desc.addUntracked<unsigned int>("multipleEntriesForLumi", 0);
    desc.addUntracked<bool>("declareLast", false);
    desc.addUntracked<bool>("declareAllLast", false);
    descriptions.add("source", desc);
  }

  edm::InputSource::ItemTypeInfo SourceWithWaits::getNextItemType() {
    constexpr unsigned int secondsToMicroseconds = 1000000;

    if (startedNewRun_) {
      usleep(secondsToMicroseconds * sleepAfterStartOfRun_);
      startedNewRun_ = false;
    }

    if (lastEventOfLumi_ || noEventsInLumi_) {
      usleep(secondsToMicroseconds * timePerLumi_ / (eventsPerLumi_[currentLumi_ - 1] + 1));
      lastEventOfLumi_ = false;
      noEventsInLumi_ = false;
    }

    // First three cases are for the initial file, run, and lumi transitions
    // Note that there will always be at exactly one file and at least
    // one run from this test source.
    if (currentFile_ == 0u) {
      ++currentFile_;
      return ItemType::IsFile;
    }
    // First Run
    else if (currentRun_ == 0u) {
      ++currentRun_;
      if (currentRun_ != multipleEntriesForRun_) {
        startedNewRun_ = true;
        auto const position =
            (declareLast_ || declareAllLast_) ? ItemPosition::LastItemToBeMerged : ItemPosition::NotLastItemToBeMerged;
        return ItemTypeInfo(ItemType::IsRun, position);
      } else {
        // declareAllLast_ with multipleEntriesForRun_ or multipleEntriesForLumi_ is an intentional bug, used to test
        // if the Framework detects the potential InputSource bug and throws an exception.
        auto const position = declareAllLast_ ? ItemPosition::LastItemToBeMerged : ItemPosition::NotLastItemToBeMerged;
        return ItemTypeInfo(ItemType::IsRun, position);
      }
    }
    // If configured, a second Entry for the same run number and reduced ProcessHistoryID
    else if (currentRun_ == multipleEntriesForRun_) {
      multipleEntriesForRun_ = 0;
      startedNewRun_ = true;
      auto const position =
          (declareLast_ || declareAllLast_) ? ItemPosition::LastItemToBeMerged : ItemPosition::NotLastItemToBeMerged;
      return ItemTypeInfo(ItemType::IsRun, position);
    }
    // First lumi
    else if (currentLumi_ == 0u && lumisPerRun_ != 0) {
      ++currentLumi_;
      ++lumiInCurrentRun_;
      // The job will stop when we hit the end of the eventsPerLumi vector
      // unless maxEvents stopped it earlier.
      if ((currentLumi_ - 1) >= eventsPerLumi_.size()) {
        return ItemType::IsStop;
      }
      if (currentLumi_ != multipleEntriesForLumi_) {
        if (eventsPerLumi_[currentLumi_ - 1] == 0) {
          noEventsInLumi_ = true;
        }
        auto const position =
            (declareLast_ || declareAllLast_) ? ItemPosition::LastItemToBeMerged : ItemPosition::NotLastItemToBeMerged;
        return ItemTypeInfo(ItemType::IsLumi, position);
      } else {
        // declareAllLast_ with multipleEntriesForRun_ or multipleEntriesForLumi_ is an intentional bug, used to test
        // if the Framework detects the potential InputSource bug and throws an exception.
        auto const position = declareAllLast_ ? ItemPosition::LastItemToBeMerged : ItemPosition::NotLastItemToBeMerged;
        return ItemTypeInfo(ItemType::IsLumi, position);
      }
    }
    // If configured, a second Entry for the same lumi number in the same run
    else if (currentLumi_ == multipleEntriesForLumi_ && lumisPerRun_ != 0) {
      multipleEntriesForLumi_ = 0;
      if (eventsPerLumi_[currentLumi_ - 1] == 0) {
        noEventsInLumi_ = true;
      }
      auto const position =
          (declareLast_ || declareAllLast_) ? ItemPosition::LastItemToBeMerged : ItemPosition::NotLastItemToBeMerged;
      return ItemTypeInfo(ItemType::IsLumi, position);
    }
    // Handle events in the current lumi
    else if (eventInCurrentLumi_ < eventsPerLumi_[currentLumi_ - 1] && lumisPerRun_ != 0) {
      // note the argument to usleep is microseconds, timePerLumi_ is in seconds
      usleep(secondsToMicroseconds * timePerLumi_ / (eventsPerLumi_[currentLumi_ - 1] + 1));
      ++eventInCurrentLumi_;
      ++currentEvent_;
      if (eventInCurrentLumi_ == eventsPerLumi_[currentLumi_ - 1]) {
        lastEventOfLumi_ = true;
      }
      return ItemType::IsEvent;
    }
    // Next lumi
    else if (lumiInCurrentRun_ < lumisPerRun_) {
      ++currentLumi_;
      ++lumiInCurrentRun_;
      // The job will stop when we hit the end of the eventsPerLumi vector
      // unless maxEvents stopped it earlier.
      if ((currentLumi_ - 1) >= eventsPerLumi_.size()) {
        return ItemType::IsStop;
      }
      eventInCurrentLumi_ = 0;
      if (currentLumi_ != multipleEntriesForLumi_) {
        if (eventsPerLumi_[currentLumi_ - 1] == 0) {
          noEventsInLumi_ = true;
        }
        auto const position =
            (declareLast_ || declareAllLast_) ? ItemPosition::LastItemToBeMerged : ItemPosition::NotLastItemToBeMerged;
        return ItemTypeInfo(ItemType::IsLumi, position);
      } else {
        // declareAllLast_ with multipleEntriesForRun_ or multipleEntriesForLumi_ is an intentional bug, used to test
        // if the Framework detects the potential InputSource bug and throws an exception.
        auto const position = declareAllLast_ ? ItemPosition::LastItemToBeMerged : ItemPosition::NotLastItemToBeMerged;
        return ItemTypeInfo(ItemType::IsLumi, position);
      }
    }
    // Next run
    else {
      // The job will stop when we hit the end of the eventsPerLumi vector
      // unless maxEvents stopped it earlier. Don't start the run if
      // it will end with no lumis in it.
      if (currentLumi_ >= eventsPerLumi_.size()) {
        return ItemType::IsStop;
      }
      ++currentRun_;
      // Avoid infinite job if lumisPerRun_ is 0
      if (currentRun_ > 100) {
        return ItemType::IsStop;
      }
      lumiInCurrentRun_ = 0;
      if (currentRun_ != multipleEntriesForRun_) {
        startedNewRun_ = true;
        auto const position =
            (declareLast_ || declareAllLast_) ? ItemPosition::LastItemToBeMerged : ItemPosition::NotLastItemToBeMerged;
        return ItemTypeInfo(ItemType::IsRun, position);
      } else {
        // declareAllLast_ with multipleEntriesForRun_ or multipleEntriesForLumi_ is an intentional bug, used to test
        // if the Framework detects the potential InputSource bug and throws an exception.
        auto const position = declareAllLast_ ? ItemPosition::LastItemToBeMerged : ItemPosition::NotLastItemToBeMerged;
        return ItemTypeInfo(ItemType::IsRun, position);
      }
    }
    // Should be impossible to get here
    assert(false);
    // return something so it will compile
    return ItemType::IsStop;
  }

  std::shared_ptr<edm::RunAuxiliary> SourceWithWaits::readRunAuxiliary_() {
    edm::Timestamp ts = edm::Timestamp(1);
    return std::make_shared<edm::RunAuxiliary>(currentRun_, ts, edm::Timestamp::invalidTimestamp());
  }

  std::shared_ptr<edm::LuminosityBlockAuxiliary> SourceWithWaits::readLuminosityBlockAuxiliary_() {
    edm::Timestamp ts = edm::Timestamp(1);
    return std::make_shared<edm::LuminosityBlockAuxiliary>(
        currentRun_, currentLumi_, ts, edm::Timestamp::invalidTimestamp());
  }

  void SourceWithWaits::readEvent_(edm::EventPrincipal& eventPrincipal) {
    bool isRealData = false;
    edm::EventAuxiliary aux(
        edm::EventID(currentRun_, currentLumi_, currentEvent_), processGUID(), edm::Timestamp(1), isRealData);
    auto history = processHistoryRegistry().getMapped(aux.processHistoryID());
    eventPrincipal.fillEventPrincipal(aux, history);
  }

}  // namespace edmtest
using edmtest::SourceWithWaits;
DEFINE_FWK_INPUT_SOURCE(SourceWithWaits);
