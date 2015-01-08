/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "PoolSource.h"
#include "InputFile.h"
#include "RootInputFileSequence.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/src/SharedResourcesRegistry.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputType.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include <set>

namespace edm {

  class BranchID;
  class LuminosityBlockID;
  class EventID;
  class ThinnedAssociationsHelper;

  namespace {
    void checkHistoryConsistency(Principal const& primary, Principal const& secondary) {
      ProcessHistory const& ph1 = primary.processHistory();
      ProcessHistory const& ph2 = secondary.processHistory();
      if(ph1 != ph2 && !isAncestor(ph2, ph1)) {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::checkConsistency") <<
          "The secondary file is not an ancestor of the primary file\n";
      }
    }
    void checkConsistency(EventPrincipal const& primary, EventPrincipal const& secondary) {
      if(!isSameEvent(primary, secondary)) {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::checkConsistency") <<
          primary.id() << " has inconsistent EventAuxiliary data in the primary and secondary file\n";
      }
    }
    void checkConsistency(LuminosityBlockAuxiliary const& primary, LuminosityBlockAuxiliary const& secondary) {
      if(primary.id() != secondary.id()) {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::checkConsistency") <<
          primary.id() << " has inconsistent LuminosityBlockAuxiliary data in the primary and secondary file\n";
      }
    }
    void checkConsistency(RunAuxiliary const& primary, RunAuxiliary const& secondary) {
      if(primary.id() != secondary.id()) {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::checkConsistency") <<
          primary.id() << " has inconsistent RunAuxiliary data in the primary and secondary file\n";
      }
    }
  }

  PoolSource::PoolSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    VectorInputSource(pset, desc),
    rootServiceChecker_(),
    primaryFileSequence_(new RootInputFileSequence(pset, *this, catalog(), desc.allocations_->numberOfStreams(),
                                                   primary() ? InputType::Primary : InputType::SecondarySource)),
    secondaryFileSequence_(catalog(1).empty() ? nullptr :
                           new RootInputFileSequence(pset, *this, catalog(1), desc.allocations_->numberOfStreams(),
                           InputType::SecondaryFile)),
    secondaryRunPrincipal_(),
    secondaryLumiPrincipal_(),
    secondaryEventPrincipals_(),
    branchIDsToReplace_(),
    resourceSharedWithDelayedReaderPtr_(primary()?
                                        new SharedResourcesAcquirer{SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader()}:
                                        static_cast<SharedResourcesAcquirer*>(nullptr))
  {
    if(secondaryFileSequence_) {
      unsigned int nStreams = desc.allocations_->numberOfStreams();
      assert(primary());
      secondaryEventPrincipals_.reserve(nStreams);
      for(unsigned int index = 0; index < nStreams; ++index) {
        secondaryEventPrincipals_.emplace_back(new EventPrincipal(secondaryFileSequence_->fileProductRegistry(),
                                                                  secondaryFileSequence_->fileBranchIDListHelper(),
                                                                  std::make_shared<ThinnedAssociationsHelper const>(),
                                                                  processConfiguration(),
                                                                  nullptr,
                                                                  index));
      }
      std::array<std::set<BranchID>, NumBranchTypes> idsToReplace;
      ProductRegistry::ProductList const& secondary = secondaryFileSequence_->fileProductRegistry()->productList();
      ProductRegistry::ProductList const& primary = primaryFileSequence_->fileProductRegistry()->productList();
      std::set<BranchID> associationsFromSecondary;
      typedef ProductRegistry::ProductList::const_iterator const_iterator;
      typedef ProductRegistry::ProductList::iterator iterator;
      //this is the registry used by the 'outside' world and only has the primary file information in it at present
      ProductRegistry::ProductList& fullList = productRegistryUpdate().productListUpdator();
      for(const_iterator it = secondary.begin(), itEnd = secondary.end(); it != itEnd; ++it) {
        if(it->second.present()) {
          idsToReplace[it->second.branchType()].insert(it->second.branchID());
          if(it->second.branchType() == InEvent &&
             it->second.unwrappedType() == typeid(ThinnedAssociation)) {
            associationsFromSecondary.insert(it->second.branchID());
          }
          //now make sure this is marked as not dropped else the product will not be 'get'table from the Event
          iterator itFound = fullList.find(it->first);
          if(itFound != fullList.end()) {
            itFound->second.setDropped(false);
          }
        }
      }
      for(const_iterator it = primary.begin(), itEnd = primary.end(); it != itEnd; ++it) {
        if(it->second.present()) {
          idsToReplace[it->second.branchType()].erase(it->second.branchID());
          associationsFromSecondary.erase(it->second.branchID());
        }
      }
      if(idsToReplace[InEvent].empty() && idsToReplace[InLumi].empty() && idsToReplace[InRun].empty()) {
        secondaryFileSequence_.reset();
      } else {
        for(int i = InEvent; i < NumBranchTypes; ++i) {
          branchIDsToReplace_[i].reserve(idsToReplace[i].size());
          for(std::set<BranchID>::const_iterator it = idsToReplace[i].begin(), itEnd = idsToReplace[i].end();
               it != itEnd; ++it) {
            branchIDsToReplace_[i].push_back(*it);
          }
        }
      }
      secondaryFileSequence_->initAssociationsFromSecondary(associationsFromSecondary);
    }
  }

  PoolSource::~PoolSource() {}

  void
  PoolSource::endJob() {
    if(secondaryFileSequence_) secondaryFileSequence_->endJob();
    primaryFileSequence_->endJob();
    InputFile::reportReadBranches();
  }

  std::unique_ptr<FileBlock>
  PoolSource::readFile_() {
    std::unique_ptr<FileBlock> fb = primaryFileSequence_->readFile_();
    if(secondaryFileSequence_) {
      fb->setNotFastClonable(FileBlock::HasSecondaryFileSequence);
    }
    return std::move(fb);
  }

  void PoolSource::closeFile_() {
    primaryFileSequence_->closeFile_();
  }

  std::shared_ptr<RunAuxiliary>
  PoolSource::readRunAuxiliary_() {
    return primaryFileSequence_->readRunAuxiliary_();
  }

  std::shared_ptr<LuminosityBlockAuxiliary>
  PoolSource::readLuminosityBlockAuxiliary_() {
    return primaryFileSequence_->readLuminosityBlockAuxiliary_();
  }

  void
  PoolSource::readRun_(RunPrincipal& runPrincipal) {
    primaryFileSequence_->readRun_(runPrincipal);
    if(secondaryFileSequence_ && !branchIDsToReplace_[InRun].empty()) {
      bool found = secondaryFileSequence_->skipToItem(runPrincipal.run(), 0U, 0U);
      if(found) {
        std::shared_ptr<RunAuxiliary> secondaryAuxiliary = secondaryFileSequence_->readRunAuxiliary_();
        checkConsistency(runPrincipal.aux(), *secondaryAuxiliary);
        secondaryRunPrincipal_.reset(new RunPrincipal(secondaryAuxiliary,
                                                      secondaryFileSequence_->fileProductRegistry(),
                                                      processConfiguration(),
                                                      nullptr,
                                                      runPrincipal.index()));
        secondaryFileSequence_->readRun_(*secondaryRunPrincipal_);
        checkHistoryConsistency(runPrincipal, *secondaryRunPrincipal_);
        runPrincipal.recombine(*secondaryRunPrincipal_, branchIDsToReplace_[InRun]);
      } else {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::readRun_")
          << " Run " << runPrincipal.run()
          << " is not found in the secondary input files\n";
      }
    }
  }

  void
  PoolSource::readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal) {
    primaryFileSequence_->readLuminosityBlock_(lumiPrincipal);
    if(secondaryFileSequence_ && !branchIDsToReplace_[InLumi].empty()) {
      bool found = secondaryFileSequence_->skipToItem(lumiPrincipal.run(), lumiPrincipal.luminosityBlock(), 0U);
      if(found) {
        std::shared_ptr<LuminosityBlockAuxiliary> secondaryAuxiliary = secondaryFileSequence_->readLuminosityBlockAuxiliary_();
        checkConsistency(lumiPrincipal.aux(), *secondaryAuxiliary);
        secondaryLumiPrincipal_.reset(new LuminosityBlockPrincipal(secondaryAuxiliary,
                                                                   secondaryFileSequence_->fileProductRegistry(),
                                                                   processConfiguration(),
                                                                   nullptr,
                                                                   lumiPrincipal.index()));
        secondaryFileSequence_->readLuminosityBlock_(*secondaryLumiPrincipal_);
        checkHistoryConsistency(lumiPrincipal, *secondaryLumiPrincipal_);
        lumiPrincipal.recombine(*secondaryLumiPrincipal_, branchIDsToReplace_[InLumi]);
      } else {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::readLuminosityBlock_")
          << " Run " << lumiPrincipal.run()
          << " LuminosityBlock " << lumiPrincipal.luminosityBlock()
          << " is not found in the secondary input files\n";
      }
    }
  }

  void
  PoolSource::readEvent_(EventPrincipal& eventPrincipal) {
    primaryFileSequence_->readEvent(eventPrincipal);
    if(secondaryFileSequence_ && !branchIDsToReplace_[InEvent].empty()) {
      bool found = secondaryFileSequence_->skipToItem(eventPrincipal.run(),
                                                      eventPrincipal.luminosityBlock(),
                                                      eventPrincipal.id().event());
      if(found) {
        EventPrincipal& secondaryEventPrincipal = *secondaryEventPrincipals_[eventPrincipal.streamID().value()];
        secondaryFileSequence_->readEvent(secondaryEventPrincipal);
        checkConsistency(eventPrincipal, secondaryEventPrincipal);
        checkHistoryConsistency(eventPrincipal, secondaryEventPrincipal);
        eventPrincipal.recombine(secondaryEventPrincipal, branchIDsToReplace_[InEvent]);
        eventPrincipal.mergeProvenanceRetrievers(secondaryEventPrincipal);
        secondaryEventPrincipal.clearPrincipal();
      } else {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::readEvent_") <<
          eventPrincipal.id() << " is not found in the secondary input files\n";
      }
    }
  }

  bool
  PoolSource::readIt(EventID const& id, EventPrincipal& eventPrincipal, StreamContext& streamContext) {
    bool found = primaryFileSequence_->skipToItem(id.run(), id.luminosityBlock(), id.event());
    if(!found) return false;
    EventSourceSentry sentry(*this, streamContext);
    readEvent_(eventPrincipal);
    return true;
  }

  InputSource::ItemType
  PoolSource::getNextItemType() {
    RunNumber_t run = IndexIntoFile::invalidRun;
    LuminosityBlockNumber_t lumi = IndexIntoFile::invalidLumi;
    EventNumber_t event = IndexIntoFile::invalidEvent;
    InputSource::ItemType itemType = primaryFileSequence_->getNextItemType(run, lumi, event);
    if(secondaryFileSequence_ && (IsSynchronize != state())) {
      if(itemType == IsRun || itemType == IsLumi || itemType == IsEvent) {
        if(!secondaryFileSequence_->containedInCurrentFile(run, lumi, event)) {
          return IsSynchronize;
        }
      }
    }
    return itemType;
  }

  void
  PoolSource::preForkReleaseResources() {
    primaryFileSequence_->closeFile_();
  }
  
  SharedResourcesAcquirer*
  PoolSource::resourceSharedWithDelayedReader_() const {
    return resourceSharedWithDelayedReaderPtr_.get();
  }

  // Rewind to before the first event that was read.
  void
  PoolSource::rewind_() {
    primaryFileSequence_->rewind_();
  }

  // Advance "offset" events.  Offset can be positive or negative (or zero).
  void
  PoolSource::skip(int offset) {
    primaryFileSequence_->skipEvents(offset);
  }

  bool
  PoolSource::goToEvent_(EventID const& eventID) {
    return primaryFileSequence_->goToEvent(eventID);
  }

  void
  PoolSource::readOneRandom(EventPrincipal& cache, CLHEP::HepRandomEngine* engine) {
    assert(!secondaryFileSequence_);
    primaryFileSequence_->readOneRandom(cache, engine);
  }

  bool
  PoolSource::readOneRandomWithID(EventPrincipal& cache, LuminosityBlockID const& lumiID, CLHEP::HepRandomEngine* engine) {
    assert(!secondaryFileSequence_);
    return primaryFileSequence_->readOneRandomWithID(cache, lumiID, engine);
  }

  bool
  PoolSource::readOneSequential(EventPrincipal& cache) {
    assert(!secondaryFileSequence_);
    return primaryFileSequence_->readOneSequential(cache);
  }

  bool
  PoolSource::readOneSequentialWithID(EventPrincipal& cache, LuminosityBlockID const& lumiID) {
    assert(!secondaryFileSequence_);
    return primaryFileSequence_->readOneSequentialWithID(cache, lumiID);
  }

  void
  PoolSource::readOneSpecified(EventPrincipal& cache, EventID const& id) {
    assert(!secondaryFileSequence_);
    primaryFileSequence_->readOneSpecified(cache, id);
  }

  void
  PoolSource::dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) {
    assert(!secondaryFileSequence_);
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
