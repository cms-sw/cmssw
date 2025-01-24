#ifndef FWCore_Sources_LHEProvenanceHelper_h
#define FWCore_Sources_LHEProvenanceHelper_h

#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace lhef {
  class LHERunInfo;
}

namespace edm {
  class ProcessHistoryRegistry;
  class ProductRegistry;
  class TypeID;
  class BranchIDListHelper;
  struct LHEProvenanceHelper {
    explicit LHEProvenanceHelper(TypeID const& eventProductType,
                                 TypeID const& runProductType,
                                 ProductRegistry& productRegistry,
                                 BranchIDListHelper& helper);
    ParameterSet fillCommonProcessParameterSet();
    void lheAugment(lhef::LHERunInfo const* runInfo);
    ProcessHistoryID lheInit(ProcessHistoryRegistry& processHistoryRegistry);
    ProductDescription const eventProductProductDescription_;
    ProductDescription const runProductProductDescription_;
    ProductProvenance eventProductProvenance_;
    ParameterSet const commonProcessParameterSet_;
    ParameterSet processParameterSet_;
    BranchListIndexes branchListIndexes_;
  };
}  // namespace edm
#endif
