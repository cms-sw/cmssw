#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/Registry.h"

namespace edm {

    Event::Event(EventPrincipal& ep, ModuleDescription const& md) :
	DataViewImpl(ep, md, InEvent),
	aux_(ep.aux()),
        luminosityBlock_(ep.luminosityBlockPrincipalSharedPtr() ? new LuminosityBlock(ep.luminosityBlockPrincipal(), md) : 0),
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

    ProductID
    Event::makeProductID(ConstBranchDescription const& desc) const {
      return eventPrincipal().branchIDToProductID(desc.branchID());
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

  ProcessHistoryID const&
  Event::processHistoryID() const {
    return eventPrincipal().history().processHistoryID();
  }


  Provenance
  Event::getProvenance(BranchID const& bid) const {
    return principal().getProvenance(bid);
  }

  Provenance
  Event::getProvenance(ProductID const& pid) const {
    return eventPrincipal().getProvenance(pid);
  }

  void
  Event::getAllProvenance(std::vector<Provenance const*> & provenances) const {
    principal().getAllProvenance(provenances);
  }

  bool
  Event::getProcessParameterSet(std::string const& processName,
				ParameterSet& ps) const {
    // Get the ProcessHistory for this event.
    ProcessHistoryRegistry* phr = ProcessHistoryRegistry::instance();
    ProcessHistory ph;
    if (!phr->getMapped(processHistoryID(), ph)) {
      throw Exception(errors::NotFound) 
	<< "ProcessHistoryID " << processHistoryID()
	<< " is claimed to describe " << id()
	<< "\nbut is not found in the ProcessHistoryRegistry.\n"
	<< "This file is malformed.\n";
    }

    ProcessConfiguration config;
    bool process_found = ph.getConfigurationForProcess(processName, config);
    if (process_found) {
      pset::Registry::instance()->getMapped(config.parameterSetID(), ps);
      assert(!ps.empty());
    }
    return process_found;
  }

  BasicHandle
  Event::getByProductID_(ProductID const& oid) const {
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
	// set provenance
	std::auto_ptr<ProductProvenance> productProvenancePtr(
		new ProductProvenance(pit->second->branchID(),
				   productstatus::present(),
				   gotBranchIDVector));
	ep.put(pit->first, *pit->second, productProvenancePtr);
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
