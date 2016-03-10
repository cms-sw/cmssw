/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/ProductHolderBase.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Framework/interface/ProductDeletedException.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <cassert>

namespace edm {

  ProductHolderBase::ProductHolderBase() {}

  ProductHolderBase::~ProductHolderBase() {}

  bool
  ProductHolderBase::provenanceAvailable() const {
    // If this product is from a the current process,
    // the provenance is available if and only if a product has been put.
    if(branchDescription().produced()) {
      return productResolved();
    }
    // If this product is from a prior process, the provenance is available,
    // although the per event part may have been dropped.
    return true;
  }

  TypeID
  ProductHolderBase::productType() const {
    return TypeID(branchDescription().wrappedTypeID());
  }

  Provenance const*
  ProductHolderBase::provenance() const {
    return provenance_();
  }

  void
  ProductHolderBase::write(std::ostream& os) const {
    // This is grossly inadequate. It is also not critical for the
    // first pass.
    os << std::string("ProductHolder for product with ID: ")
       << productID();
  }
}
