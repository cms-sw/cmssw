#include <vector>

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

namespace edm {

  Run::Run(RunPrincipal& rp, ModuleDescription const& md) :
	provRecorder_(rp, md),
	aux_(rp.aux()) {
  }

  RunPrincipal &
  Run::runPrincipal() {
    return dynamic_cast<RunPrincipal &>(provRecorder_.principal());
  }

  RunPrincipal const &
  Run::runPrincipal() const {
    return dynamic_cast<RunPrincipal const&>(provRecorder_.principal());
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

  bool
  Run::getProcessParameterSet(std::string const& processName,
			      std::vector<ParameterSet>& psets) const
  {
    // Get the relevant ProcessHistoryIDs
    ProcessHistoryRegistry* phreg = ProcessHistoryRegistry::instance();
    std::vector<ProcessHistoryID> historyIDs;
    

    // Get the relevant ParameterSetIDs.
    std::vector<ParameterSetID> psetIdsUsed;
    for (std::vector<ProcessHistoryID>::const_iterator
	   i = historyIDs.begin(),
	   e = historyIDs.end();
	 i != e;
	 ++i)
      {
	ProcessHistory temp;
	phreg->getMapped(*i, temp);
      }

    // Look up the ParameterSets for these IDs.
    pset::Registry* psreg = pset::Registry::instance();
    for (std::vector<ParameterSetID>::const_iterator 
	   i = psetIdsUsed.begin(),
	   e = psetIdsUsed.end();
	 i != e;
	 ++i)
      {
	ParameterSet temp;
	psreg->getMapped(*i, temp);
	psets.push_back(temp);	  
      }

    return false;
  }

  void
  Run::commit_() {
    // fill in guts of provenance here
    RunPrincipal & rp = runPrincipal();
    ProductPtrVec::iterator pit(putProducts().begin());
    ProductPtrVec::iterator pie(putProducts().end());

    while(pit!=pie) {
	// set provenance
	std::auto_ptr<ProductProvenance> runEntryInfoPtr(
		new ProductProvenance(pit->second->branchID(),
				    productstatus::present()));
	rp.put(pit->first, *pit->second, runEntryInfoPtr);
	++pit;
    }

    // the cleanup is all or none
    putProducts().clear();
  }
  
  ProcessHistory const&
  Run::processHistory() const {
    return provRecorder_.processHistory();
  }
  

}
