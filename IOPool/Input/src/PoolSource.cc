/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "PoolSource.h"
#include "RootInputFileSequence.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "TTreeCache.h"

#include <set>

namespace edm {

  class LuminosityBlockID;
  class EventID;

  namespace {
    void checkHistoryConsistency(Principal const& primary, Principal const& secondary) {
      ProcessHistory const& ph1 = primary.processHistory();
      ProcessHistory const& ph2 = secondary.processHistory();
      if (ph1 != ph2 && !isAncestor(ph2, ph1)) {
        throw edm::Exception(errors::MismatchedInputFiles, "PoolSource::checkConsistency") <<
          "The secondary file is not an ancestor of the primary file\n";
      }
    }
    void checkConsistency(EventPrincipal const& primary, EventPrincipal const& secondary) {
      if (!isSameEvent(primary, secondary)) {
        throw edm::Exception(errors::MismatchedInputFiles, "PoolSource::checkConsistency") <<
          primary.id() << " has inconsistent EventAuxiliary data in the primary and secondary file\n";
      }
      checkHistoryConsistency(primary, secondary);
    }
    void checkConsistency(LuminosityBlockPrincipal const& primary, LuminosityBlockPrincipal const& secondary) {
      if (primary.id() != secondary.id()) {
        throw edm::Exception(errors::MismatchedInputFiles, "PoolSource::checkConsistency") <<
          primary.id() << " has inconsistent LuminosityBlockAuxiliary data in the primary and secondary file\n";
      }
      checkHistoryConsistency(primary, secondary);
    }
    void checkConsistency(RunPrincipal const& primary, RunPrincipal const& secondary) {
      if (primary.id() != secondary.id()) {
        throw edm::Exception(errors::MismatchedInputFiles, "PoolSource::checkConsistency") <<
          primary.id() << " has inconsistent RunAuxiliary data in the primary and secondary file\n";
      }
      checkHistoryConsistency(primary, secondary);
    }
  }

  PoolSource::PoolSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    VectorInputSource(pset, desc),
    rootServiceChecker_(),
    primaryFileSequence_(new RootInputFileSequence(pset, *this, catalog(), primary())),
    secondaryFileSequence_(catalog(1).empty() ? 0 : new RootInputFileSequence(pset, *this, catalog(1), false)),
    branchIDsToReplace_(),
    numberOfEventsBeforeBigSkip_(0),
    numberOfEventsInBigSkip_(0),
    numberOfSequentialEvents_(0),
    forkedChildIndex_(0)  
  {
    if (secondaryFileSequence_) {
      boost::array<std::set<BranchID>, NumBranchTypes> idsToReplace;
      ProductRegistry::ProductList const& secondary = secondaryFileSequence_->fileProductRegistry().productList();
      ProductRegistry::ProductList const& primary = primaryFileSequence_->fileProductRegistry().productList();
      typedef ProductRegistry::ProductList::const_iterator const_iterator;
      for (const_iterator it = secondary.begin(), itEnd = secondary.end(); it != itEnd; ++it) {
	if (it->second.present()) idsToReplace[it->second.branchType()].insert(it->second.branchID());
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
    if (secondaryFileSequence_) {
        boost::shared_ptr<FileBlock> fb = primaryFileSequence_->readFile_();
	fb->setNotFastClonable(FileBlock::HasSecondaryFileSequence);
        return fb;
    }
    return primaryFileSequence_->readFile_();
  }

  void PoolSource::closeFile_() {
    primaryFileSequence_->closeFile_();
  }

  boost::shared_ptr<RunPrincipal>
  PoolSource::readRun_() {
    RunSourceSentry(*this);
    if (secondaryFileSequence_ && !branchIDsToReplace_[InRun].empty()) {
      boost::shared_ptr<RunPrincipal> primaryPrincipal = primaryFileSequence_->readRun_();
      bool found = secondaryFileSequence_->skipToItem(primaryPrincipal->run(), 0U, 0U, true, false);
      if (found) {
        boost::shared_ptr<RunPrincipal> secondaryPrincipal = secondaryFileSequence_->readRun_();
        checkConsistency(*primaryPrincipal, *secondaryPrincipal);      
        primaryPrincipal->recombine(*secondaryPrincipal, branchIDsToReplace_[InRun]);
      } else {
        throw edm::Exception(errors::MismatchedInputFiles, "PoolSource::readRun_")
          << " Run " << primaryPrincipal->run()
          << " is not found in the secondary input files\n";
      }
      return primaryPrincipal;
    }
    return primaryFileSequence_->readRun_();
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  PoolSource::readLuminosityBlock_() {
    LumiSourceSentry(*this);
    if (secondaryFileSequence_ && !branchIDsToReplace_[InLumi].empty()) {
      boost::shared_ptr<LuminosityBlockPrincipal> primaryPrincipal = primaryFileSequence_->readLuminosityBlock_();
      bool found = secondaryFileSequence_->skipToItem(primaryPrincipal->run(), primaryPrincipal->luminosityBlock(), 0U, true, false);
      if (found) {
        boost::shared_ptr<LuminosityBlockPrincipal> secondaryPrincipal = secondaryFileSequence_->readLuminosityBlock_();
        checkConsistency(*primaryPrincipal, *secondaryPrincipal);      
        primaryPrincipal->recombine(*secondaryPrincipal, branchIDsToReplace_[InLumi]);
      } else {
        throw edm::Exception(errors::MismatchedInputFiles, "PoolSource::readLuminosityBlock_")
          << " Run " << primaryPrincipal->run()
          << " LuminosityBlock " << primaryPrincipal->luminosityBlock()
          << " is not found in the secondary input files\n";
      }
      return primaryPrincipal;
    }
    return primaryFileSequence_->readLuminosityBlock_();
  }

  std::auto_ptr<EventPrincipal>
  PoolSource::readEvent_() {
    EventSourceSentry(*this);
    if (secondaryFileSequence_ && !branchIDsToReplace_[InEvent].empty()) {
      std::auto_ptr<EventPrincipal> primaryPrincipal = primaryFileSequence_->readEvent_();
      bool found = secondaryFileSequence_->skipToItem(primaryPrincipal->run(),
						       primaryPrincipal->luminosityBlock(),
						       primaryPrincipal->id().event(),
						       true, false);
      if (found) {
        std::auto_ptr<EventPrincipal> secondaryPrincipal = secondaryFileSequence_->readEvent_();
        checkConsistency(*primaryPrincipal, *secondaryPrincipal);      
        primaryPrincipal->recombine(*secondaryPrincipal, branchIDsToReplace_[InEvent]);
      } else {
        throw edm::Exception(errors::MismatchedInputFiles, "PoolSource::readEvent_") <<
          primaryPrincipal->id() << " is not found in the secondary input files\n";
      }
      return primaryPrincipal;
    }
    return primaryFileSequence_->readEvent_();
  }

  std::auto_ptr<EventPrincipal>
  PoolSource::readIt(EventID const& id) {
    primaryFileSequence_->skipToItem(id.run(), 0U, id.event(), false, true);
    return readEvent_();
  }

  InputSource::ItemType
  PoolSource::getNextItemType() {
    InputSource::ItemType returnValue = primaryFileSequence_->getNextItemType();
    if(returnValue == InputSource::IsEvent && 0 != numberOfEventsInBigSkip_ && 0 == --numberOfEventsBeforeBigSkip_) {
      primaryFileSequence_->skipEvents(numberOfEventsInBigSkip_);
      numberOfEventsBeforeBigSkip_ = numberOfSequentialEvents_+1;
      returnValue = primaryFileSequence_->getNextItemType();
    }    
    return returnValue;
  }

  void 
  PoolSource::postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren, unsigned int iNumberOfSequentialEvents) {
    numberOfEventsInBigSkip_ = iNumberOfSequentialEvents*(iNumberOfChildren-1);
    numberOfEventsBeforeBigSkip_ = iNumberOfSequentialEvents ;
    forkedChildIndex_ = iChildIndex;
    numberOfSequentialEvents_ = iNumberOfSequentialEvents;
    primaryFileSequence_->reset();
    rewind();
  }
   
  // Rewind to before the first event that was read.
  void
  PoolSource::rewind_() {
    primaryFileSequence_->rewind_();
    unsigned int numberToSkip = numberOfSequentialEvents_*forkedChildIndex_;
    if(0!=numberToSkip) {
      numberOfEventsBeforeBigSkip_ = numberOfSequentialEvents_ ;
      if(numberOfEventsBeforeBigSkip_ < numberToSkip) {
        numberOfEventsBeforeBigSkip_ = numberToSkip+1;
      }
      primaryFileSequence_->skipEvents(numberToSkip);
    }
    numberOfEventsBeforeBigSkip_ = numberOfSequentialEvents_+1 ;
  }

  // Advance "offset" events.  Offset can be positive or negative (or zero).
  void
  PoolSource::skip(int offset) {
    primaryFileSequence_->skipEvents(offset);
  }

  void
  PoolSource::readMany_(int number, EventPrincipalVector& result) {
    assert (!secondaryFileSequence_);
    primaryFileSequence_->readMany_(number, result);
  }

  void
  PoolSource::readMany_(int number, EventPrincipalVector& result, EventID const& id, unsigned int fileSeqNumber) {
    assert (!secondaryFileSequence_);
    primaryFileSequence_->readMany_(number, result, id, fileSeqNumber);
  }

  void
  PoolSource::readManyRandom_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) {
    assert (!secondaryFileSequence_);
    primaryFileSequence_->readManyRandom_(number, result, fileSeqNumber);
  }

  void
  PoolSource::readManySequential_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) {
    assert (!secondaryFileSequence_);
    primaryFileSequence_->readManySequential_(number, result, fileSeqNumber);
  }

  void
  PoolSource::dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) {
    assert (!secondaryFileSequence_);
    primaryFileSequence_->dropUnwantedBranches_(wantedBranches);
  }

  void
  PoolSource::fillDescriptions(ConfigurationDescriptions & descriptions) {

    edm::ParameterSetDescription desc;

    desc.addOptionalUntracked<unsigned int>("firstRun", 1U);
    desc.addOptionalUntracked<unsigned int>("firstLuminosityBlock", 1U);
    desc.addOptionalUntracked<unsigned int>("firstEvent", 1U);
    desc.addOptionalUntracked<unsigned int>("skipEvents", 0U);

    std::vector<LuminosityBlockRange> defaultLumis;
    desc.addOptionalUntracked<std::vector<LuminosityBlockRange> >("lumisToSkip", defaultLumis);
    desc.addOptionalUntracked<std::vector<LuminosityBlockRange> >("lumisToProcess", defaultLumis);

    std::vector<EventRange> defaultEvents;
    desc.addOptionalUntracked<std::vector<EventRange> >("eventsToSkip", defaultEvents);
    desc.addOptionalUntracked<std::vector<EventRange> >("eventsToProcess", defaultEvents);

    desc.addOptionalUntracked<bool>("noEventSort", false);
    desc.addOptionalUntracked<bool>("skipBadFiles", false);
    desc.addOptionalUntracked<bool>("needSecondaryFileNames", false);
    desc.addOptionalUntracked<bool>("dropDescendantsOfDroppedBranches", true);
    desc.addOptionalUntracked<unsigned int>("cacheSize", 0U);
    desc.addOptionalUntracked<int>("treeMaxVirtualSize", -1);
    desc.addOptionalUntracked<unsigned int>("setRunNumber", 0U);

    std::vector<std::string> defaultStrings(1U, std::string("keep *"));
    desc.addOptionalUntracked<std::vector<std::string> >("inputCommands", defaultStrings);

    std::string defaultString("permissive");
    desc.addOptionalUntracked<std::string>("fileMatchMode", defaultString);

    defaultString = "checkAllFilesOpened";
    desc.addOptionalUntracked<std::string>("duplicateCheckMode", defaultString);

    defaultStrings.clear();
    desc.addUntracked<std::vector<std::string> >("fileNames", defaultStrings);
    desc.addOptionalUntracked<std::vector<std::string> >("secondaryFileNames", defaultStrings);

    defaultString.clear();
    desc.addOptionalUntracked<std::string>("overrideCatalog", defaultString);

    defaultString = "RunsLumisAndEvents";
    desc.addOptionalUntracked<std::string>("processingMode", defaultString);

    descriptions.add("source", desc);
  }
}
