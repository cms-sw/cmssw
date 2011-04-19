#include "FWCore/Framework/interface/RunPrincipal.h"

#include "DataFormats/Common/interface/WrapperHolder.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/Group.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  RunPrincipal::RunPrincipal(
    boost::shared_ptr<RunAuxiliary> aux,
    boost::shared_ptr<ProductRegistry const> reg,
    ProcessConfiguration const& pc) :
      Base(reg, pc, InRun),
      aux_(aux) {
  }

  void
  RunPrincipal::fillRunPrincipal(
    boost::shared_ptr<BranchMapper> mapper,
    boost::shared_ptr<DelayedReader> rtrv) {
    if(productRegistry().anyProductProduced()) {
      checkProcessHistory();
    }
    fillPrincipal(aux_->processHistoryID(), mapper, rtrv);
    if(productRegistry().anyProductProduced()) {
      addToProcessHistory();
    }
    mapper->processHistoryID() = processHistoryID();
    for (const_iterator i = this->begin(), iEnd = this->end(); i != iEnd; ++i) {
      (*i)->setProvenance(mapper);
    }
  }

  void 
  RunPrincipal::put(
	ConstBranchDescription const& bd,
	WrapperHolder const& edp,
	std::auto_ptr<ProductProvenance> productProvenance) {

    assert(bd.produced());
    if(!edp.isValid()) {
      throw edm::Exception(edm::errors::InsertFailure,"Null Pointer")
	<< "put: Cannot put because auto_ptr to product is null."
	<< "\n";
    }
    branchMapperPtr()->insert(*productProvenance);
    Group *g = getExistingGroup(bd.branchID());
    assert(g);
    // Group assumes ownership
    putOrMerge(edp, productProvenance, g);
  }

  void
  RunPrincipal::readImmediate() const {
    for (Principal::const_iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
      Group const& g = **i;
      if(!g.branchDescription().produced()) {
        if(!g.productUnavailable()) {
          resolveProductImmediate(g);
        }
      }
    }
    branchMapperPtr()->setDelayedRead(false);
  }

  void
  RunPrincipal::resolveProductImmediate(Group const& g) const {
    if(g.branchDescription().produced()) return; // nothing to do.

    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(g.branchDescription());
    WrapperHolder edp(store()->getProduct(bk, g.productData().getInterface(), this));

    // Now fix up the Group
    if(edp.isValid()) {
      putOrMerge(edp, &g);
    }
  }

  void
  RunPrincipal::checkProcessHistory() const {
    ProcessHistory ph;
    ProcessHistoryRegistry::instance()->getMapped(aux_->processHistoryID(), ph);
    std::string const& processName = processConfiguration().processName();
    for (ProcessHistory::const_iterator it = ph.begin(), itEnd = ph.end(); it != itEnd; ++it) {
      if(processName == it->processName()) {
	throw edm::Exception(errors::Configuration, "Duplicate Process")
	  << "The process name " << processName << " was previously used on these products.\n"
	  << "Please modify the configuration file to use a distinct process name.\n";
      }
    }
  }

  void
  RunPrincipal::addToProcessHistory() {
    ProcessHistory& ph = processHistoryUpdate();
    ph.push_back(processConfiguration());
    //OPTIMIZATION NOTE:  As of 0_9_0_pre3
    // For very simple Sources (e.g. EmptySource) this routine takes up nearly 50% of the time per event.
    // 96% of the time for this routine is being spent in computing the
    // ProcessHistory id which happens because we are reconstructing the ProcessHistory for each event.
    // (The process ID is first computed in the call to 'insertMapped(..)' below.)
    // It would probably be better to move the ProcessHistory construction out to somewhere
    // which persists for longer than one Event

    ProcessHistoryRegistry::instance()->insertMapped(ph);
    setProcessHistory(*this);
  }

  void
  RunPrincipal::swap(RunPrincipal& iOther) {
    swapBase(iOther);
    std::swap(aux_, iOther.aux_);
  }
}
