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

#include "TTreeCache.h"

#include <set>

namespace edm {
  namespace {
    void checkConsistency(EventPrincipal const& primary, EventPrincipal const& secondary) {
      if (!isSameEvent(primary, secondary)) {
        throw cms::Exception("Inconsistent Data", "PoolSource::checkConsistency") <<
          primary.id() << " has inconsistent EventAuxiliary data in the primary and secondary file\n";
      }
      ProcessHistory const& ph1 = primary.processHistory();
      ProcessHistory const& ph2 = secondary.processHistory();
      if (ph1 != ph2 && !isAncestor(ph2, ph1)) {
        throw cms::Exception("Inconsistent Data", "PoolSource::checkConsistency") <<
          "For " << primary.id() << " , the secondary file is not an ancestor  of the primary file\n";
      }
    }
  }

  PoolSource::PoolSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    VectorInputSource(pset, desc),
    primaryFileSequence_(new RootInputFileSequence(pset, *this, catalog(), primary())),
    secondaryFileSequence_(catalog(1).empty() ? 0 : new RootInputFileSequence(pset, *this, catalog(1), false)),
    branchIDsToReplace_() {
    if (secondaryFileSequence_) {
      std::set<BranchID> idsToReplace;
      ProductRegistry::ProductList const& secondary = secondaryFileSequence_->fileProductRegistry().productList();
      ProductRegistry::ProductList const& primary = primaryFileSequence_->fileProductRegistry().productList();
      typedef ProductRegistry::ProductList::const_iterator const_iterator;
      for (const_iterator it = secondary.begin(), itEnd = secondary.end(); it != itEnd; ++it) {
	if (it->second.present() && it->second.branchType() == InEvent) idsToReplace.insert(it->second.branchID());
      }
      for (const_iterator it = primary.begin(), itEnd = primary.end(); it != itEnd; ++it) {
	if (it->second.present() && it->second.branchType() == InEvent) idsToReplace.erase(it->second.branchID());
      }
      if (idsToReplace.empty()) {
        secondaryFileSequence_.reset();
      } else {
        branchIDsToReplace_.reserve(idsToReplace.size());
	for (std::set<BranchID>::const_iterator it = idsToReplace.begin(), itEnd = idsToReplace.end();
	     it != itEnd; ++it) {
	  branchIDsToReplace_.push_back(*it);
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
    return primaryFileSequence_->readRun_();
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  PoolSource::readLuminosityBlock_() {
    return primaryFileSequence_->readLuminosityBlock_();
  }

  std::auto_ptr<EventPrincipal>
  PoolSource::readEvent_() {
    if (secondaryFileSequence_) {
      std::auto_ptr<EventPrincipal> primaryPrincipal = primaryFileSequence_->readEvent_();
      std::auto_ptr<EventPrincipal> secondaryPrincipal = secondaryFileSequence_->readIt(primaryPrincipal->id(), primaryPrincipal->luminosityBlock(), true);
      if (secondaryPrincipal.get() != 0) {
        checkConsistency(*primaryPrincipal, *secondaryPrincipal);      
        primaryPrincipal->recombine(*secondaryPrincipal, branchIDsToReplace_);
      } else {
        throw edm::Exception(errors::NotFound, "PoolSource::readEvent_") <<
          primaryPrincipal->id() << " is not found in the secondary input file\n";
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
        primaryPrincipal->recombine(*secondaryPrincipal, branchIDsToReplace_);
      } else {
        throw edm::Exception(errors::NotFound, "PoolSource::readIt") <<
          primaryPrincipal->id() << " is not found in the secondary input file\n";
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

