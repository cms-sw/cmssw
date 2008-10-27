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

#include "TTreeCache.h"

#include <set>

namespace edm {
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
    branchIDsToReplace_() {
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
    if (secondaryFileSequence_) secondaryFileSequence_->closeFile_();
    closeFile_();
  }

  boost::shared_ptr<FileBlock>
  PoolSource::readFile_() {
    if (secondaryFileSequence_) {
        boost::shared_ptr<FileBlock> fb = primaryFileSequence_->readFile_();
	fb->setNotFastCopyable();
        return fb;
    }
    return primaryFileSequence_->readFile_();
  }

  void PoolSource::closeFile_() {
    primaryFileSequence_->closeFile_();
  }

  boost::shared_ptr<RunPrincipal>
  PoolSource::readRun_() {
    if (secondaryFileSequence_ && !branchIDsToReplace_[InRun].empty()) {
      boost::shared_ptr<RunPrincipal> primaryPrincipal = primaryFileSequence_->readRun_();
      boost::shared_ptr<RunPrincipal> secondaryPrincipal = secondaryFileSequence_->readIt(primaryPrincipal->id());
      if (secondaryPrincipal.get() != 0) {
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
    if (secondaryFileSequence_ && !branchIDsToReplace_[InLumi].empty()) {
      boost::shared_ptr<LuminosityBlockPrincipal> primaryPrincipal = primaryFileSequence_->readLuminosityBlock_();
      boost::shared_ptr<LuminosityBlockPrincipal> secondaryPrincipal = secondaryFileSequence_->readIt(primaryPrincipal->id());
      if (secondaryPrincipal.get() != 0) {
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
    if (secondaryFileSequence_ && !branchIDsToReplace_[InEvent].empty()) {
      std::auto_ptr<EventPrincipal> primaryPrincipal = primaryFileSequence_->readEvent_();
      std::auto_ptr<EventPrincipal> secondaryPrincipal = secondaryFileSequence_->readIt(primaryPrincipal->id(), primaryPrincipal->luminosityBlock(), true);
      if (secondaryPrincipal.get() != 0) {
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
    if (secondaryFileSequence_) {
      std::auto_ptr<EventPrincipal> primaryPrincipal = primaryFileSequence_->readIt(id);
      std::auto_ptr<EventPrincipal> secondaryPrincipal = secondaryFileSequence_->readIt(id, primaryPrincipal->luminosityBlock(), true);
      if (secondaryPrincipal.get() != 0) {
        checkConsistency(*primaryPrincipal, *secondaryPrincipal);      
        primaryPrincipal->recombine(*secondaryPrincipal, branchIDsToReplace_[InEvent]);
      } else {
        throw edm::Exception(errors::MismatchedInputFiles, "PoolSource::readIt") <<
          primaryPrincipal->id() << " is not found in the secondary input files\n";
      }
      return primaryPrincipal;
    }
    return primaryFileSequence_->readIt(id);
  }

  InputSource::ItemType
  PoolSource::getNextItemType() {
    return primaryFileSequence_->getNextItemType();
  }

  // Rewind to before the first event that was read.
  void
  PoolSource::rewind_() {
    primaryFileSequence_->rewind_();
  }

  // Advance "offset" events.  Offset can be positive or negative (or zero).
  void
  PoolSource::skip(int offset) {
    primaryFileSequence_->skip(offset);
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
  PoolSource::dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) {
    assert (!secondaryFileSequence_);
    assert (!primary());
    primaryFileSequence_->dropUnwantedBranches_(wantedBranches);
  }
}

