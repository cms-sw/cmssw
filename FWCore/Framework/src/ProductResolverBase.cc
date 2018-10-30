/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/ProductResolverBase.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Framework/interface/ProductDeletedException.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <cassert>

namespace edm {

  ProductResolverBase::ProductResolverBase() {}

  ProductResolverBase::~ProductResolverBase() {}

  bool
  ProductResolverBase::provenanceAvailable() const {
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
  ProductResolverBase::productType() const {
    return TypeID(branchDescription().wrappedTypeID());
  }

  Provenance const*
  ProductResolverBase::provenance() const {
    return provenance_();
  }

  void
  ProductResolverBase::retrieveAndMerge_(Principal const&, MergeableRunProductMetadata const*) const {
  }

  void
  ProductResolverBase::setMergeableRunProductMetadata_(MergeableRunProductMetadata const*) {
  }

  void
  ProductResolverBase::write(std::ostream& os) const {
    // This is grossly inadequate. It is also not critical for the
    // first pass.
    os << std::string("ProductResolver for product with ID: ")
       << productID();
  }
  
  void
  ProductResolverBase::setupUnscheduled(UnscheduledConfigurator const&) {}
  
}
