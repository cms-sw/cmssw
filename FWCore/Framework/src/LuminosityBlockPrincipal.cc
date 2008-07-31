#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Group.h"

namespace edm {

  LuminosityBlockPrincipal::LuminosityBlockPrincipal(LuminosityBlockAuxiliary const& aux,
	boost::shared_ptr<ProductRegistry const> reg,
	ProcessConfiguration const& pc,
	ProcessHistoryID const& hist,
	boost::shared_ptr<Mapper> mapper, 
	boost::shared_ptr<DelayedReader> rtrv) :
	  Base(reg, pc, hist, mapper, rtrv),
	  runPrincipal_(),
	  aux_(aux) {}

  void
  LuminosityBlockPrincipal::addOrReplaceGroup(std::auto_ptr<Group> g) {

    Group* group = getExistingGroup(*g);
    if (group != 0) {

      if (!group->productUnavailable()) {
        assert(group->product() != 0);
      }
      if (!g->productUnavailable()) {
        assert(g->product() != 0);
      }

      group->mergeGroup(g.get());
    } else {
      addGroup_(g);
    }
  }

  void
  LuminosityBlockPrincipal::addGroup(ConstBranchDescription const& bd) {
    std::auto_ptr<Group> g(new Group(bd));
    addOrReplaceGroup(g);
  }

  void
  LuminosityBlockPrincipal::addGroup(std::auto_ptr<EDProduct> prod,
	ConstBranchDescription const& bd,
	std::auto_ptr<RunLumiEntryInfo> entryInfo) {
    std::auto_ptr<Group> g(new Group(prod, bd, entryInfo));
    addOrReplaceGroup(g);
  }

  void
  LuminosityBlockPrincipal::addGroup(ConstBranchDescription const& bd,
	std::auto_ptr<RunLumiEntryInfo> entryInfo) {
    std::auto_ptr<Group> g(new Group(bd, entryInfo));
    addOrReplaceGroup(g);
  }

  void 
  LuminosityBlockPrincipal::put(std::auto_ptr<EDProduct> edp,
		ConstBranchDescription const& bd,
		std::auto_ptr<RunLumiEntryInfo> entryInfo) {

    if (edp.get() == 0) {
      throw edm::Exception(edm::errors::InsertFailure,"Null Pointer")
	<< "put: Cannot put because auto_ptr to product is null."
	<< "\n";
    }
    this->addToProcessHistory();
    branchMapperPtr()->insert(*entryInfo);
    // Group assumes ownership
    this->addGroup(edp, bd, entryInfo);
  }

  Provenance
  LuminosityBlockPrincipal::getProvenance(BranchID const& bid) const {
    SharedConstGroupPtr const& g = getGroup(bid, false, true);
    if (g.get() == 0) {
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	<< "getProvenance: no product with given branch id: "<< bid << "\n";
    }

    if (g->onDemand()) {
      unscheduledFill(g->productDescription().moduleLabel());
    }
    // We already tried to produce the unscheduled products above
    // If they still are not there, then throw
    if (g->onDemand()) {
      throw edm::Exception(edm::errors::ProductNotFound)
	<< "getProvenance: no product with given BranchID: "<< bid <<"\n";
    }

    return *g->provenance();
  }

  // This one is mostly for test printout purposes
  // No attempt to trigger on demand execution
  // Skips provenance when the EDProduct is not there
  void
  LuminosityBlockPrincipal::getAllProvenance(std::vector<Provenance const*> & provenances) const {
    provenances.clear();
    for (Base::const_iterator i = begin(), iEnd = end(); i != iEnd; ++i) {
      if (i->second->provenanceAvailable()) {
        resolveProvenance(*i->second);
        if (i->second->provenance()->branchEntryInfoSharedPtr() &&
            i->second->provenance()->isPresent() &&
            i->second->provenance()->product().present())
           provenances.push_back(i->second->provenance());
        }
    }
  }

  void
  LuminosityBlockPrincipal::resolveProvenance(Group const& g) const {
    if (!g.entryInfoPtr()) {
      // Now fix up the Group
      g.setProvenance(branchMapperPtr()->branchToEntryInfo(g.productDescription(). branchID()));
    }
  }

  void
  LuminosityBlockPrincipal::mergeLuminosityBlock(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {

    aux_.mergeAuxiliary(lbp->aux());

    for (Base::const_iterator i = lbp->begin(), iEnd = lbp->end(); i != iEnd; ++i) {
 
      std::auto_ptr<Group> g(new Group());
      g->swap(*i->second);

      addOrReplaceGroup(g);
    }
  }
}

