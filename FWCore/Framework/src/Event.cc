#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

    Event::Event(EventPrincipal& ep, ModuleDescription const& md) :
	DataViewImpl<EventEntryInfo>(ep, md, InEvent),
	aux_(ep.aux()),
	luminosityBlock_(new LuminosityBlock(ep.luminosityBlockPrincipal(), md)),
	gotProductIDs_(),
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
    // fill in guts of provenance here
    EventPrincipal & ep = eventPrincipal();
    ProductPtrVec::iterator pit(putProducts().begin());
    ProductPtrVec::iterator pie(putProducts().end());

    std::vector<ProductID> gotProductIDVector;
    if (!gotProductIDs_.empty()) {
      gotProductIDVector.reserve(gotProductIDs_.size());
      for (ProductIDSet::const_iterator it = gotProductIDs_.begin(), itEnd = gotProductIDs_.end();
	  it != itEnd; ++it) {
        gotProductIDVector.push_back(*it);
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
				    gotProductIDVector));
	ep.put(pr, *pit->second, eventEntryInfoPtr);
	++pit;
    }

    // the cleanup is all or none
    putProducts().clear();
  }

}
