#include "FWCore/Framework/interface/Run.h"

#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/ProductPutterBase.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

namespace edm {

  std::string const Run::emptyString_;

  Run::Run(RunTransitionInfo const& info, ModuleDescription const& md, ModuleCallingContext const* mcc, bool isAtEnd)
      : Run(info.principal(), md, mcc, isAtEnd) {}

  Run::Run(RunPrincipal const& rp,
           ModuleDescription const& md,
           ModuleCallingContext const* moduleCallingContext,
           bool isAtEnd)
      : provRecorder_(rp, md, isAtEnd), aux_(rp.aux()), moduleCallingContext_(moduleCallingContext) {}

  Run::~Run() {}

  RunAuxiliary const& Run::runAuxiliary() const { return aux_; }

  Run::CacheIdentifier_t Run::cacheIdentifier() const { return runPrincipal().cacheIdentifier(); }

  RunIndex Run::index() const { return runPrincipal().index(); }

  RunPrincipal const& Run::runPrincipal() const { return dynamic_cast<RunPrincipal const&>(provRecorder_.principal()); }

  Provenance const& Run::getProvenance(BranchID const& bid) const { return runPrincipal().getProvenance(bid); }

  StableProvenance const& Run::getStableProvenance(BranchID const& bid) const {
    return runPrincipal().getStableProvenance(bid);
  }

  void Run::getAllStableProvenance(std::vector<StableProvenance const*>& provenances) const {
    runPrincipal().getAllStableProvenance(provenances);
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

  void Run::setProducer(ProducerBase const* iProducer) {
    provRecorder_.setProducer(iProducer);
    //set appropriate size
    putProducts_.resize(provRecorder_.putTokenIndexToProductResolverIndex().size());
  }

  void Run::commit_(std::vector<edm::ProductResolverIndex> const& iShouldPut) {
    RunPrincipal const& rp = runPrincipal();
    size_t nPut = 0;
    for (size_t i = 0; i < putProducts().size(); ++i) {
      auto& p = get_underlying_safe(putProducts()[i]);
      if (p) {
        rp.put(provRecorder_.putTokenIndexToProductResolverIndex()[i], std::move(p));
        ++nPut;
      }
    }

    auto sz = iShouldPut.size();
    if (sz != 0 and sz != nPut) {
      //some were missed
      auto& p = provRecorder_.principal();
      for (auto index : iShouldPut) {
        auto resolver = p.getProductResolverByIndex(index);
        if (not resolver->productResolved() and isEndTransition(provRecorder_.transition()) ==
                                                    resolver->branchDescription().availableOnlyAtEndTransition()) {
          dynamic_cast<ProductPutterBase const*>(resolver)->putProduct(std::unique_ptr<WrapperBase>());
        }
      }
    }

    // the cleanup is all or none
    putProducts().clear();
  }

  ProcessHistoryID const& Run::processHistoryID() const { return runPrincipal().processHistoryID(); }

  ProcessHistory const& Run::processHistory() const { return provRecorder_.processHistory(); }

  BasicHandle Run::getByLabelImpl(std::type_info const&,
                                  std::type_info const& iProductType,
                                  const InputTag& iTag) const {
    BasicHandle h = provRecorder_.getByLabel_(TypeID(iProductType), iTag, moduleCallingContext_);
    return h;
  }
}  // namespace edm
