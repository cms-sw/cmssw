#ifndef FWCore_Sources_LHEProvenanceHelper_h
#define FWCore_Sources_LHEProvenanceHelper_h

#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace lhef {
  class LHERunInfo;
}

namespace edm {
  class ProductRegistry;
  class TypeID;
  struct LHEProvenanceHelper {
    explicit LHEProvenanceHelper(TypeID const& eventProductType, TypeID const& runProductType);
    void lheAugment(lhef::LHERunInfo const* runInfo);
    ProcessHistoryID lheInit(ProductRegistry& productRegistry);
    ConstBranchDescription eventProductBranchDescription_;
    ConstBranchDescription runProductBranchDescription_;
    ProductProvenance eventProductProvenance_;
    ParameterSet processParameterSet_;
  };
}
#endif
