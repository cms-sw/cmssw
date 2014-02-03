#include "FWCore/Framework/interface/Run.h"

#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Utilities/interface/Algorithms.h"

namespace edm {

  std::string const Run::emptyString_;

  Run::Run(RunPrincipal& rp, ModuleDescription const& md,
           ModuleCallingContext const* moduleCallingContext) :
        provRecorder_(rp, md),
        aux_(rp.aux()),
        moduleCallingContext_(moduleCallingContext)  {
  }

  Run::~Run() {
    // anything left here must be the result of a failure
    // let's record them as failed attempts in the event principal
    for_all(putProducts_, principal_get_adapter_detail::deleter());
  }

  RunIndex Run::index() const { return runPrincipal().index();}
  
  RunPrincipal&
  Run::runPrincipal() {
    return dynamic_cast<RunPrincipal&>(provRecorder_.principal());
  }

  RunPrincipal const&
  Run::runPrincipal() const {
    return dynamic_cast<RunPrincipal const&>(provRecorder_.principal());
  }

  Provenance
  Run::getProvenance(BranchID const& bid) const {
    return runPrincipal().getProvenance(bid, moduleCallingContext_);
  }

  void
  Run::getAllProvenance(std::vector<Provenance const*>& provenances) const {
    runPrincipal().getAllProvenance(provenances);
  }

/* Not yet fully implemented
  bool
  Run::getProcessParameterSet(std::string const& processName, std::vector<ParameterSet>& psets) const {
    // Get the relevant ProcessHistoryIDs
    ProcessHistoryRegistry* phreg = ProcessHistoryRegistry::instance();
    // Need to fill these in.
    std::vector<ProcessHistoryID> historyIDs;


    // Get the relevant ParameterSetIDs.
    // Need to fill these in.
    std::vector<ParameterSetID> psetIdsUsed;
    for(std::vector<ProcessHistoryID>::const_iterator
           i = historyIDs.begin(),
           e = historyIDs.end();
         i != e;
         ++i) {
      ProcessHistory temp;
      phreg->getMapped(*i, temp);
    }

    // Look up the ParameterSets for these IDs.
    pset::Registry* psreg = pset::Registry::instance();
    for(std::vector<ParameterSetID>::const_iterator
           i = psetIdsUsed.begin(),
           e = psetIdsUsed.end();
         i != e;
         ++i) {
      ParameterSet temp;
      psreg->getMapped(*i, temp);
      psets.push_back(temp);
    }

    return false;
  }
*/

  void
  Run::commit_() {
    RunPrincipal& rp = runPrincipal();
    ProductPtrVec::iterator pit(putProducts().begin());
    ProductPtrVec::iterator pie(putProducts().end());

    while(pit != pie) {
        rp.put(*pit->second, pit->first);
        // Ownership has passed, so clear the pointer.
        pit->first.reset();
        ++pit;
    }

    // the cleanup is all or none
    putProducts().clear();
  }

  ProcessHistoryID const&
  Run::processHistoryID() const {
    return runPrincipal().processHistoryID();
  }

  ProcessHistory const&
  Run::processHistory() const {
    return provRecorder_.processHistory();
  }

  void
  Run::addToGotBranchIDs(Provenance const& prov) const {
    gotBranchIDs_.insert(prov.branchID());
  }

  BasicHandle
  Run::getByLabelImpl(std::type_info const&, std::type_info const& iProductType, const InputTag& iTag) const {
    BasicHandle h = provRecorder_.getByLabel_(TypeID(iProductType), iTag, moduleCallingContext_);
    if(h.isValid()) {
      addToGotBranchIDs(*(h.provenance()));
    }
    return h;
  }
}
