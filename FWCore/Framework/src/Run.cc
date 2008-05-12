#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/RunPrincipal.h"

namespace edm {

  Run::Run(RunPrincipal& rp, ModuleDescription const& md) :
	DataViewImpl<RunLumiEntryInfo>(rp, md, InRun),
	aux_(rp.aux()) {
  }

  RunPrincipal &
  Run::runPrincipal() {
    return dynamic_cast<RunPrincipal &>(principal());
  }

  RunPrincipal const &
  Run::runPrincipal() const {
    return dynamic_cast<RunPrincipal const&>(principal());
  }

  Provenance
  Run::getProvenance(BranchID const& bid) const
  {
    return runPrincipal().getProvenance(bid);
  }

  void
  Run::getAllProvenance(std::vector<Provenance const*> & provenances) const
  {
    runPrincipal().getAllProvenance(provenances);
  }


  void
  Run::commit_() {
    // fill in guts of provenance here
    RunPrincipal & rp = runPrincipal();
    ProductPtrVec::iterator pit(putProducts().begin());
    ProductPtrVec::iterator pie(putProducts().end());

    while(pit!=pie) {
	std::auto_ptr<EDProduct> pr(pit->first);
	// note: ownership has been passed - so clear the pointer!
	pit->first = 0;

	// set provenance
	std::auto_ptr<RunLumiEntryInfo> runEntryInfoPtr(
		new RunLumiEntryInfo(pit->second->branchID(),
				    productstatus::present(),
				    pit->second->moduleDescriptionID()));
	rp.put(pr, *pit->second, runEntryInfoPtr);
	++pit;
    }

    // the cleanup is all or none
    putProducts().clear();
  }

}
