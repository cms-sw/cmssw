/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "PoolSource.h"
#include "RootInputFileSequence.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/MessageReceiverForSource.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <set>

namespace edm {

  class LuminosityBlockID;
  class EventID;

  namespace {
    void checkHistoryConsistency(Principal const& primary, Principal const& secondary) {
      ProcessHistory const& ph1 = primary.processHistory();
      ProcessHistory const& ph2 = secondary.processHistory();
      if (ph1 != ph2 && !isAncestor(ph2, ph1)) {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::checkConsistency") <<
          "The secondary file is not an ancestor of the primary file\n";
      }
    }
    void checkConsistency(EventPrincipal const& primary, EventPrincipal const& secondary) {
      if (!isSameEvent(primary, secondary)) {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::checkConsistency") <<
          primary.id() << " has inconsistent EventAuxiliary data in the primary and secondary file\n";
      }
    }
    void checkConsistency(LuminosityBlockAuxiliary const& primary, LuminosityBlockAuxiliary const& secondary) {
      if (primary.id() != secondary.id()) {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::checkConsistency") <<
          primary.id() << " has inconsistent LuminosityBlockAuxiliary data in the primary and secondary file\n";
      }
    }
    void checkConsistency(RunAuxiliary const& primary, RunAuxiliary const& secondary) {
      if (primary.id() != secondary.id()) {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::checkConsistency") <<
          primary.id() << " has inconsistent RunAuxiliary data in the primary and secondary file\n";
      }
    }
  }

  PoolSource::PoolSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    VectorInputSource(pset, desc),
    rootServiceChecker_(),
    primaryFileSequence_(new RootInputFileSequence(pset, *this, catalog(), principalCache(), primary())),
    secondaryFileSequence_(catalog(1).empty() ? 0 :
                           new RootInputFileSequence(pset, *this, catalog(1), principalCache(), false)),
    secondaryRunPrincipal_(),
    secondaryLumiPrincipal_(),
    secondaryEventPrincipal_(secondaryFileSequence_ ? new EventPrincipal(secondaryFileSequence_->fileProductRegistry(), processConfiguration()) : 0),
    branchIDsToReplace_(),
    numberOfEventsBeforeBigSkip_(0)
  {
    if (secondaryFileSequence_) {
      boost::array<std::set<BranchID>, NumBranchTypes> idsToReplace;
      ProductRegistry::ProductList const& secondary = secondaryFileSequence_->fileProductRegistry()->productList();
      ProductRegistry::ProductList const& primary = primaryFileSequence_->fileProductRegistry()->productList();
      typedef ProductRegistry::ProductList::const_iterator const_iterator;
      typedef ProductRegistry::ProductList::iterator iterator;
      //this is the registry used by the 'outside' world and only has the primary file information in it at present
      ProductRegistry::ProductList& fullList = productRegistryUpdate().productListUpdator();
      for (const_iterator it = secondary.begin(), itEnd = secondary.end(); it != itEnd; ++it) {
        if (it->second.present()) {
          idsToReplace[it->second.branchType()].insert(it->second.branchID());
          //now make sure this is marked as not dropped else the product will not be 'get'table from the Event
          iterator itFound = fullList.find(it->first);
          if(itFound != fullList.end()) {
            itFound->second.dropped()=false;
          }
        }
      }
      for (const_iterator it = primary.begin(), itEnd = primary.end(); it != itEnd; ++it) {
        if (it->second.present()) idsToReplace[it->second.branchType()].erase(it->second.branchID());
      }
      if (idsToReplace[InEvent].empty() && idsToReplace[InLumi].empty() && idsToReplace[InRun].empty()) {
        secondaryFileSequence_.reset();
      }
      else {
        for (int i = InEvent; i < NumBranchTypes; ++i) {
          branchIDsToReplace_[i].reserve(idsToReplace[i].size());
          for (std::set<BranchID>::const_iterator it = idsToReplace[i].begin(), itEnd = idsToReplace[i].end();
               it != itEnd; ++it) {
            branchIDsToReplace_[i].push_back(*it);
          }
        }
      }
    }
  }

  PoolSource::~PoolSource() {}

  void
  PoolSource::endJob() {
    if (secondaryFileSequence_) secondaryFileSequence_->endJob();
    primaryFileSequence_->endJob();
  }

  boost::shared_ptr<FileBlock>
  PoolSource::readFile_() {
    boost::shared_ptr<FileBlock> fb = primaryFileSequence_->readFile_(principalCache());
    if (secondaryFileSequence_) {
      fb->setNotFastClonable(FileBlock::HasSecondaryFileSequence);
    }
    return fb;
  }

  void PoolSource::closeFile_() {
    primaryFileSequence_->closeFile_();
  }

  boost::shared_ptr<RunAuxiliary>
  PoolSource::readRunAuxiliary_() {
    return primaryFileSequence_->readRunAuxiliary_();
  }

  boost::shared_ptr<LuminosityBlockAuxiliary>
  PoolSource::readLuminosityBlockAuxiliary_() {
    return primaryFileSequence_->readLuminosityBlockAuxiliary_();
  }

  boost::shared_ptr<RunPrincipal>
  PoolSource::readRun_(boost::shared_ptr<RunPrincipal> rpCache) {
    if (secondaryFileSequence_ && !branchIDsToReplace_[InRun].empty()) {
      boost::shared_ptr<RunPrincipal> primaryPrincipal = primaryFileSequence_->readRun_(rpCache);
      bool found = secondaryFileSequence_->skipToItem(primaryPrincipal->run(), 0U, 0U);
      if (found) {
        boost::shared_ptr<RunAuxiliary> secondaryAuxiliary = secondaryFileSequence_->readRunAuxiliary_();
        checkConsistency(primaryPrincipal->aux(), *secondaryAuxiliary);
        boost::shared_ptr<RunPrincipal> rp(new RunPrincipal(secondaryAuxiliary, secondaryFileSequence_->fileProductRegistry(), processConfiguration()));
        secondaryRunPrincipal_ = secondaryFileSequence_->readRun_(rp);
        checkHistoryConsistency(*primaryPrincipal, *secondaryRunPrincipal_);
        primaryPrincipal->recombine(*secondaryRunPrincipal_, branchIDsToReplace_[InRun]);
      } else {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::readRun_")
          << " Run " << primaryPrincipal->run()
          << " is not found in the secondary input files\n";
      }
      return primaryPrincipal;
    }
    return primaryFileSequence_->readRun_(rpCache);
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  PoolSource::readLuminosityBlock_(boost::shared_ptr<LuminosityBlockPrincipal> lbCache) {
    if (secondaryFileSequence_ && !branchIDsToReplace_[InLumi].empty()) {
      boost::shared_ptr<LuminosityBlockPrincipal> primaryPrincipal = primaryFileSequence_->readLuminosityBlock_(lbCache);
      bool found = secondaryFileSequence_->skipToItem(primaryPrincipal->run(), primaryPrincipal->luminosityBlock(), 0U);
      if (found) {
        boost::shared_ptr<LuminosityBlockAuxiliary> secondaryAuxiliary = secondaryFileSequence_->readLuminosityBlockAuxiliary_();
        checkConsistency(primaryPrincipal->aux(), *secondaryAuxiliary);
        boost::shared_ptr<LuminosityBlockPrincipal> lbp(new LuminosityBlockPrincipal(secondaryAuxiliary, secondaryFileSequence_->fileProductRegistry(), processConfiguration(), secondaryRunPrincipal_));
        secondaryLumiPrincipal_ = secondaryFileSequence_->readLuminosityBlock_(lbp);
        checkHistoryConsistency(*primaryPrincipal, *secondaryLumiPrincipal_);
        primaryPrincipal->recombine(*secondaryLumiPrincipal_, branchIDsToReplace_[InLumi]);
      } else {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::readLuminosityBlock_")
          << " Run " << primaryPrincipal->run()
          << " LuminosityBlock " << primaryPrincipal->luminosityBlock()
          << " is not found in the secondary input files\n";
      }
      return primaryPrincipal;
    }
    return primaryFileSequence_->readLuminosityBlock_(lbCache);
  }

  EventPrincipal*
  PoolSource::readEvent_() {
    EventSourceSentry(*this);
    EventPrincipal* primaryPrincipal = primaryFileSequence_->readEvent(*eventPrincipalCache(), luminosityBlockPrincipal());
    if (secondaryFileSequence_ && !branchIDsToReplace_[InEvent].empty()) {
      bool found = secondaryFileSequence_->skipToItem(primaryPrincipal->run(),
                                                      primaryPrincipal->luminosityBlock(),
                                                      primaryPrincipal->id().event());
      if (found) {
        EventPrincipal* secondaryPrincipal = secondaryFileSequence_->readEvent(*secondaryEventPrincipal_, secondaryLumiPrincipal_);
        checkConsistency(*primaryPrincipal, *secondaryPrincipal);
        checkHistoryConsistency(*primaryPrincipal, *secondaryPrincipal);
        primaryPrincipal->recombine(*secondaryPrincipal, branchIDsToReplace_[InEvent]);
        secondaryEventPrincipal_->clearPrincipal();
      } else {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::readEvent_") <<
          primaryPrincipal->id() << " is not found in the secondary input files\n";
      }
    }
    if(receiver_) {
      --numberOfEventsBeforeBigSkip_;
    }
    return primaryPrincipal;
  }

  EventPrincipal*
  PoolSource::readIt(EventID const& id) {
    bool found = primaryFileSequence_->skipToItem(id.run(), id.luminosityBlock(), id.event());
    if (!found) return 0;
    return readEvent_();
  }

  InputSource::ItemType
  PoolSource::getNextItemType() {
    if(receiver_ &&
       0 == numberOfEventsBeforeBigSkip_) {
      receiver_->receive();
      unsigned long toSkip = receiver_->numberToSkip();
      if (0 != toSkip) {
        primaryFileSequence_->skipEvents(toSkip, principalCache());
        decreaseRemainingEventsBy(toSkip);
      }
      numberOfEventsBeforeBigSkip_ = receiver_->numberOfConsecutiveIndices();
      if (0 == numberOfEventsBeforeBigSkip_ or 0==remainingEvents()) {
        return IsStop;
      }
    }
    return primaryFileSequence_->getNextItemType();;
  }

  void
  PoolSource::preForkReleaseResources() {
    primaryFileSequence_->closeFile_();
  }

  void
  PoolSource::postForkReacquireResources(boost::shared_ptr<edm::multicore::MessageReceiverForSource> iReceiver) {
    receiver_ = iReceiver;
    receiver_->receive();
    primaryFileSequence_->reset(principalCache());
    rewind();
    decreaseRemainingEventsBy(receiver_->numberToSkip());
  }

  // Rewind to before the first event that was read.
  void
  PoolSource::rewind_() {
    primaryFileSequence_->rewind_();
    if (receiver_) {
      unsigned int numberToSkip = receiver_->numberToSkip();
      if(0 != numberToSkip) {
        primaryFileSequence_->skipEvents(numberToSkip, principalCache());
      }
      numberOfEventsBeforeBigSkip_ = receiver_->numberOfConsecutiveIndices();
    }

  }

  // Advance "offset" events.  Offset can be positive or negative (or zero).
  void
  PoolSource::skip(int offset) {
    primaryFileSequence_->skipEvents(offset, principalCache());
  }

  bool
  PoolSource::goToEvent_(EventID const& eventID) {
    return primaryFileSequence_->goToEvent(eventID);
  }

  void
  PoolSource::readMany_(int number, EventPrincipalVector& result) {
    assert (!secondaryFileSequence_);
    primaryFileSequence_->readMany(number, result);
  }

  void
  PoolSource::readManyRandom_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) {
    assert (!secondaryFileSequence_);
    primaryFileSequence_->readManyRandom(number, result, fileSeqNumber);
  }

  void
  PoolSource::readManySequential_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) {
    assert (!secondaryFileSequence_);
    primaryFileSequence_->readManySequential(number, result, fileSeqNumber);
  }

  void
  PoolSource::readManySpecified_(std::vector<EventID> const& events, EventPrincipalVector& result) {
    assert (!secondaryFileSequence_);
    primaryFileSequence_->readManySpecified(events, result);
  }

  void
  PoolSource::dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) {
    assert (!secondaryFileSequence_);
    primaryFileSequence_->dropUnwantedBranches_(wantedBranches);
  }

  void
  PoolSource::fillDescriptions(ConfigurationDescriptions & descriptions) {

    ParameterSetDescription desc;

    desc.setComment("Reads EDM/Root files.");
    VectorInputSource::fillDescription(desc);
    RootInputFileSequence::fillDescription(desc);

    descriptions.add("source", desc);
  }

  bool
  PoolSource::randomAccess_() const {
    return true;
  }

  ProcessingController::ForwardState
  PoolSource::forwardState_() const {
    return primaryFileSequence_->forwardState();
  }

  ProcessingController::ReverseState
  PoolSource::reverseState_() const {
    return primaryFileSequence_->reverseState();
  }
}
