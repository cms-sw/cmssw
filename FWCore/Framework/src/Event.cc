#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  namespace {
    LuminosityBlock * newLumi(EventPrincipal& ep, ModuleDescription const& md) {
      return (ep.luminosityBlockPrincipalSharedPtr() ? new LuminosityBlock(ep.luminosityBlockPrincipal(), md) : 0);
    }
  }
    Event::Event(EventPrincipal& ep, ModuleDescription const& md) :
	DataViewImpl<EventEntryInfo>(ep, md, InEvent),
	aux_(ep.aux()),
	luminosityBlock_(newLumi(ep, md)),
	gotBranchIDs_(),
	gotViews_() {
    }

    EventPrincipal &
    Event::eventPrincipal() {
      return dynamic_cast<EventPrincipal &>(principal());
    }

    EventPrincipal const &
    Event::eventPrincipal() const {
      return dynamic_cast<EventPrincipal const&>(principal());
    }

    Run const&
    Event::getRun() const {
      return getLuminosityBlock().getRun();
    }

//   History const& 
//   Event::history() const {
//     DataViewImpl const& dvi = me();
//     EDProductGetter const* pg = dvi.prodGetter(); // certain to be non-null
//     assert(pg);
//     EventPrincipal const& ep = dynamic_cast<EventPrincipal const&>(*pg);
//     return ep.history();
//   }
  History const&
  Event::history() const {
    return eventPrincipal().history();
  }


  Provenance
  Event::getProvenance(BranchID const& bid) const
  {
    return eventPrincipal().getProvenance(bid);
  }

  Provenance
  Event::getProvenance(ProductID const& pid) const
  {
    return eventPrincipal().getProvenance(pid);
  }

  void
  Event::getAllProvenance(std::vector<Provenance const*> & provenances) const
  {
    eventPrincipal().getAllProvenance(provenances);
  }

  BasicHandle
  Event::getByProductID_(ProductID const& oid) const
  {
    return eventPrincipal().getByProductID(oid);
  }


  void
  Event::commit_() {
    commit_aux(putProducts(), true);
    commit_aux(putProductsWithoutParents(), false);
  }

  void
  Event::commit_aux(Base::ProductPtrVec& products, bool record_parents) {
    // fill in guts of provenance here
    EventPrincipal & ep = eventPrincipal();

    ProductPtrVec::iterator pit(products.begin());
    ProductPtrVec::iterator pie(products.end());

    std::vector<BranchID> gotBranchIDVector;

    // Note that gotBranchIDVector will remain empty if
    // record_parents is false (and may be empty if record_parents is
    // true).

    if (record_parents && !gotBranchIDs_.empty()) {
      gotBranchIDVector.reserve(gotBranchIDs_.size());
      for (BranchIDSet::const_iterator it = gotBranchIDs_.begin(), itEnd = gotBranchIDs_.end();
	  it != itEnd; ++it) {
        gotBranchIDVector.push_back(*it);
      }
    }

    while(pit!=pie) {
	std::auto_ptr<EDProduct> pr(pit->first);
	// note: ownership has been passed - so clear the pointer!
	pit->first = 0;

	// set provenance
	std::auto_ptr<EventEntryInfo> eventEntryInfoPtr(
		new EventEntryInfo(pit->second->branchID(),
				   productstatus::present(),
				   pit->second->moduleDescriptionID(),
				   pit->second->productIDtoAssign(),
				   gotBranchIDVector));
	ep.put(pr, *pit->second, eventEntryInfoPtr);
	++pit;
    }

    // the cleanup is all or none
    products.clear();
  }

  void
  Event::addToGotBranchIDs(Provenance const& prov) const {
    if (prov.branchDescription().transient()) {
      // If the product retrieved is transient, don't use its branch ID.
      // use the branch ID's of its parents.
      std::vector<BranchID> const& bids = prov.parents();
      for (std::vector<BranchID>::const_iterator it = bids.begin(), itEnd = bids.end(); it != itEnd; ++it) {
        gotBranchIDs_.insert(*it);
      }
    } else {
      gotBranchIDs_.insert(prov.branchID());
    }
  }

}
