#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/Run.h"

namespace edm {

  namespace {
    Run *
    newRun(LuminosityBlockPrincipal& lbp, ModuleDescription const& md) {
      return (lbp.runPrincipalSharedPtr() ? new Run(lbp.runPrincipal(), md) : 0);
    }
  }

  LuminosityBlock::LuminosityBlock(LuminosityBlockPrincipal& lbp, ModuleDescription const& md) :
	DataViewImpl<RunLumiEntryInfo>(lbp, md, InLumi),
	aux_(lbp.aux()),
	run_(newRun(lbp, md)) {
  }

  LuminosityBlockPrincipal &
  LuminosityBlock::luminosityBlockPrincipal() {
    return dynamic_cast<LuminosityBlockPrincipal &>(principal());
  }

  LuminosityBlockPrincipal const &
  LuminosityBlock::luminosityBlockPrincipal() const {
    return dynamic_cast<LuminosityBlockPrincipal const&>(principal());
  }

  Provenance
  LuminosityBlock::getProvenance(BranchID const& bid) const
  {
    return luminosityBlockPrincipal().getProvenance(bid);
  }

  void
  LuminosityBlock::getAllProvenance(std::vector<Provenance const*> & provenances) const
  {
    luminosityBlockPrincipal().getAllProvenance(provenances);
  }


  void
  LuminosityBlock::commit_() {
    // fill in guts of provenance here
    LuminosityBlockPrincipal & lbp = luminosityBlockPrincipal();
    ProductPtrVec::iterator pit(putProducts().begin());
    ProductPtrVec::iterator pie(putProducts().end());

    while(pit!=pie) {
	std::auto_ptr<EDProduct> pr(pit->first);
	// note: ownership has been passed - so clear the pointer!
	pit->first = 0;

	// set provenance
	std::auto_ptr<RunLumiEntryInfo> lumiEntryInfoPtr(
		new RunLumiEntryInfo(pit->second->branchID(),
				    productstatus::present(),
				    pit->second->moduleDescriptionID()));
	lbp.put(pr, *pit->second, lumiEntryInfoPtr);
	++pit;
    }

    // the cleanup is all or none
    putProducts().clear();
  }

}
