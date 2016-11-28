/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "PoolSource.h"
#include "InputFile.h"
#include "RootPrimaryFileSequence.h"
#include "RootSecondaryFileSequence.h"
#include "RunHelper.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
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
    InputSource(pset, desc),
    rootServiceChecker_(),
    catalog_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames"),
      pset.getUntrackedParameter<std::string>("overrideCatalog", std::string())),
    secondaryCatalog_(pset.getUntrackedParameter<std::vector<std::string> >("secondaryFileNames", std::vector<std::string>()),
      pset.getUntrackedParameter<std::string>("overrideCatalog", std::string())),
    secondaryRunPrincipal_(),
    secondaryLumiPrincipal_(),
    secondaryEventPrincipals_(),
    branchIDsToReplace_(),
    nStreams_(desc.allocations_->numberOfStreams()),
    skipBadFiles_(pset.getUntrackedParameter<bool>("skipBadFiles")),
    bypassVersionCheck_(pset.getUntrackedParameter<bool>("bypassVersionCheck")),
    treeMaxVirtualSize_(pset.getUntrackedParameter<int>("treeMaxVirtualSize")),
    productSelectorRules_(pset, "inputCommands", "InputSource"),
    dropDescendants_(pset.getUntrackedParameter<bool>("dropDescendantsOfDroppedBranches")),
    labelRawDataLikeMC_(pset.getUntrackedParameter<bool>("labelRawDataLikeMC")),
    runHelper_(makeRunHelper(pset)),
    resourceSharedWithDelayedReaderPtr_(),
    // Note: primaryFileSequence_ and secondaryFileSequence_ need to be initialized last, because they use data members
    // initialized previously in their own initialization.
    primaryFileSequence_(new RootPrimaryFileSequence(pset, *this, catalog_)),
    secondaryFileSequence_(secondaryCatalog_.empty() ? nullptr :
                           new RootSecondaryFileSequence(pset, *this, secondaryCatalog_))
  {
    auto resources = SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader();
    resourceSharedWithDelayedReaderPtr_ = std::make_unique<SharedResourcesAcquirer>(std::move(resources.first));
    mutexSharedWithDelayedReader_ = resources.second;
    
    if (secondaryCatalog_.empty() && pset.getUntrackedParameter<bool>("needSecondaryFileNames", false)) {
      throw Exception(errors::Configuration, "PoolSource") << "'secondaryFileNames' must be specified\n";
    }
    if(secondaryFileSequence_) {
      secondaryEventPrincipals_.reserve(nStreams_);
      for(unsigned int index = 0; index < nStreams_; ++index) {
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
      //this is the registry used by the 'outside' world and only has the primary file information in it at present
      ProductRegistry::ProductList& fullList = productRegistryUpdate().productListUpdator();
      for(auto const& item : secondary) {
        if(item.second.present()) {
          idsToReplace[item.second.branchType()].insert(item.second.branchID());
          if(item.second.branchType() == InEvent &&
             item.second.unwrappedType() == typeid(ThinnedAssociation)) {
            associationsFromSecondary.insert(item.second.branchID());
          }
          //now make sure this is marked as not dropped else the product will not be 'get'table from the Event
          auto itFound = fullList.find(item.first);
          if(itFound != fullList.end()) {
            itFound->second.setDropped(false);
          }
        }
      }
      for(auto const& item : primary) {
        if(item.second.present()) {
          idsToReplace[item.second.branchType()].erase(item.second.branchID());
          associationsFromSecondary.erase(item.second.branchID());
        }
      }
      if(idsToReplace[InEvent].empty() && idsToReplace[InLumi].empty() && idsToReplace[InRun].empty()) {
        secondaryFileSequence_ = nullptr; // propagate_const<T> has no reset() function
      } else {
        for(int i = InEvent; i < NumBranchTypes; ++i) {
          branchIDsToReplace_[i].reserve(idsToReplace[i].size());
          for(auto const& id : idsToReplace[i]) {
            branchIDsToReplace_[i].push_back(id);
          }
        }
        secondaryFileSequence_->initAssociationsFromSecondary(associationsFromSecondary);
      }
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
    return fb;
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
        secondaryRunPrincipal_ = std::make_shared<RunPrincipal>(secondaryAuxiliary,
                                                      secondaryFileSequence_->fileProductRegistry(),
                                                      processConfiguration(),
                                                      nullptr,
                                                      runPrincipal.index());
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
        secondaryLumiPrincipal_ = std::make_shared<LuminosityBlockPrincipal>(secondaryAuxiliary,
                                                                   secondaryFileSequence_->fileProductRegistry(),
                                                                   processConfiguration(),
                                                                   nullptr,
                                                                   lumiPrincipal.index());
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
    return runHelper_->nextItemType(state(), itemType);
  }

  void
  PoolSource::preForkReleaseResources() {
    primaryFileSequence_->closeFile_();
  }

  std::pair<SharedResourcesAcquirer*,std::recursive_mutex*>
  PoolSource::resourceSharedWithDelayedReader_() {
    return std::make_pair(resourceSharedWithDelayedReaderPtr_.get(), mutexSharedWithDelayedReader_.get());
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
  PoolSource::fillDescriptions(ConfigurationDescriptions & descriptions) {

    ParameterSetDescription desc;

    std::vector<std::string> defaultStrings;
    desc.setComment("Reads EDM/Root files.");
    desc.addUntracked<std::vector<std::string> >("fileNames")
        ->setComment("Names of files to be processed.");
    desc.addUntracked<std::vector<std::string> >("secondaryFileNames", defaultStrings)
        ->setComment("Names of secondary files to be processed.");
    desc.addUntracked<bool>("needSecondaryFileNames", false)
        ->setComment("If True, 'secondaryFileNames' must be specified and be non-empty.");
    desc.addUntracked<std::string>("overrideCatalog", std::string());
    desc.addUntracked<bool>("skipBadFiles", false)
        ->setComment("True:  Ignore any missing or unopenable input file.\n"
                     "False: Throw exception if missing or unopenable input file.");
    desc.addUntracked<bool>("bypassVersionCheck", false)
        ->setComment("True:  Bypass release version check.\n"
                     "False: Throw exception if reading file in a release prior to the release in which the file was written.");
    desc.addUntracked<int>("treeMaxVirtualSize", -1)
        ->setComment("Size of ROOT TTree TBasket cache. Affects performance.");
    desc.addUntracked<bool>("dropDescendantsOfDroppedBranches", true)
        ->setComment("If True, also drop on input any descendent of any branch dropped on input.");
    desc.addUntracked<bool>("labelRawDataLikeMC", true)
        ->setComment("If True: replace module label for raw data to match MC. Also use 'LHC' as process.");
    ProductSelectorRules::fillDescription(desc, "inputCommands");
    InputSource::fillDescription(desc);
    RootPrimaryFileSequence::fillDescription(desc);
    RunHelperBase::fillDescription(desc);

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
