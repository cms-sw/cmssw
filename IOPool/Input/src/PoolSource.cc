/*----------------------------------------------------------------------
$Id: PoolSource.cc,v 1.77 2008/01/08 06:57:39 wmtan Exp $
----------------------------------------------------------------------*/
#include "PoolSource.h"
#include "RootInputFileSequence.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "IOPool/Common/interface/ClassFiller.h"

namespace edm {
  PoolSource::PoolSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    VectorInputSource(pset, desc),
    primaryFileSequence_(new RootInputFileSequence(pset, *this, catalog())),
    secondaryFileSequence_(catalog(1).empty() ? 0 : new RootInputFileSequence(pset, *this, catalog(1))) {
    ClassFiller();
  }

  PoolSource::~PoolSource() {}

  void
  PoolSource::endJob() {
    closeFile_();
  }

  boost::shared_ptr<FileBlock>
  PoolSource::readFile_() {
    return(primaryFileSequence_->readFile_());
  }

  void PoolSource::closeFile_() {
    if (secondaryFileSequence_) secondaryFileSequence_->closeFile_();
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
  PoolSource::readEvent_(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    return primaryFileSequence_->readEvent_(lbp);
  }

  std::auto_ptr<EventPrincipal>
  PoolSource::readIt(EventID const& id) {
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
    primaryFileSequence_->readMany_(number, result);
  }

  void
  PoolSource::readMany_(int number, EventPrincipalVector& result, EventID const& id, unsigned int fileSeqNumber) {
    primaryFileSequence_->readMany_(number, result, id, fileSeqNumber);
  }

  void
  PoolSource::readManyRandom_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) {
    primaryFileSequence_->readManyRandom_(number, result, fileSeqNumber);
  }
}

