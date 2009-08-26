#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/Run.h"

namespace edm {

  LuminosityBlock::LuminosityBlock(LuminosityBlockPrincipal& lbp, ModuleDescription const& md) :
	provRecorder_(lbp, md),
	aux_(lbp.aux()),
	run_(lbp.runPrincipalSharedPtr() ? new Run(lbp.runPrincipal(), md) : 0) {
  }

  LuminosityBlockPrincipal &
  LuminosityBlock::luminosityBlockPrincipal() {
    return dynamic_cast<LuminosityBlockPrincipal &>(provRecorder_.principal());
  }

  LuminosityBlockPrincipal const &
  LuminosityBlock::luminosityBlockPrincipal() const {
    return dynamic_cast<LuminosityBlockPrincipal const&>(provRecorder_.principal());
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
	// set provenance
	std::auto_ptr<ProductProvenance> lumiEntryInfoPtr(
		new ProductProvenance(pit->second->branchID(),
				    productstatus::present()));
	lbp.put(pit->first, *pit->second, lumiEntryInfoPtr);
	++pit;
    }

    // the cleanup is all or none
    putProducts().clear();
  }

  ProcessHistory const&
  LuminosityBlock::processHistory() const {
    return provRecorder_.processHistory();
  }
  
}
