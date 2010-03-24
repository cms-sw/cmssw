#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/Run.h"

namespace edm {

  LuminosityBlock::LuminosityBlock(LuminosityBlockPrincipal& lbp, ModuleDescription const& md) :
	provRecorder_(lbp, md),
	aux_(lbp.aux()),
	run_(new Run(lbp.runPrincipal(), md)) {
  }

  LuminosityBlock::~LuminosityBlock() {
    // anything left here must be the result of a failure
    // let's record them as failed attempts in the event principal
    for_all(putProducts_, principal_get_adapter_detail::deleter());
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
        std::auto_ptr<EDProduct> pr(pit->first);
        // Ownership has passed, so clear the pointer.
        pit->first = 0;
	lbp.put(*pit->second, pr, lumiEntryInfoPtr);
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
