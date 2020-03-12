#ifndef FWCore_Sources_LHEProvenanceHelper_h
#define FWCore_Sources_LHEProvenanceHelper_h

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace lhef {
  class LHERunInfo;
}

namespace edm {
  class ProcessHistoryRegistry;
  class ProductRegistry;
  class TypeID;
  struct LHEProvenanceHelper {
    explicit LHEProvenanceHelper(TypeID const& eventProductType,
                                 TypeID const& runProductType,
                                 ProductRegistry& productRegistry);
    ParameterSet fillCommonProcessParameterSet();
    void lheAugment(lhef::LHERunInfo const* runInfo);
    ProcessHistoryID lheInit(ProcessHistoryRegistry& processHistoryRegistry);
    BranchDescription const eventProductBranchDescription_;
    BranchDescription const runProductBranchDescription_;
    ProductProvenance eventProductProvenance_;
    ParameterSet const commonProcessParameterSet_;
    ParameterSet processParameterSet_;
  };
}  // namespace edm
#endif
