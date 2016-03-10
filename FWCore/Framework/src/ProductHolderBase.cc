/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/ProductHolderBase.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Framework/interface/ProductDeletedException.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <cassert>

namespace edm {

  void ProductHolderBase::throwProductDeletedException() const {
    ProductDeletedException exception;
    exception << "ProductHolderBase::resolveProduct_: The product matching all criteria was already deleted\n"
      << "Looking for type: " << branchDescription().unwrappedTypeID() << "\n"
      << "Looking for module label: " << moduleLabel() << "\n"
      << "Looking for productInstanceName: " << productInstanceName() << "\n"
      << (processName().empty() ? "" : "Looking for process: ") << processName() << "\n"
      << "This means there is a configuration error.\n"
      << "The module which is asking for this data must be configured to state that it will read this data.";
    throw exception;
    
  }
  
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

  void
  ProductHolderBase::putOrMergeProduct(std::unique_ptr<WrapperBase> prod) const {
    if(not prod) {return;}
    if(putOrMergeProduct_()) {
      putProduct(std::move(prod));
    } else {
      mergeProduct(std::move(prod));
    }

  }
  
  TypeID
  ProductHolderBase::productType() const {
    return TypeID(branchDescription().wrappedTypeID());
  }

  void
  ProductHolderBase::reallyCheckType(WrapperBase const& prod) const {
    // Check if the types match.
    TypeID typeID(prod.dynamicTypeInfo());
    if(typeID != branchDescription().unwrappedTypeID()) {
      // Types do not match.
      throw Exception(errors::EventCorruption)
          << "Product on branch " << branchDescription().branchName() << " is of wrong type.\n"
          << "It is supposed to be of type " << branchDescription().className() << ".\n"
          << "It is actually of type " << typeID.className() << ".\n";
    }
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
