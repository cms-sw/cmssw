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

  void
  ProductHolderBase::mergeTheProduct(std::unique_ptr<WrapperBase> edp) const {
    if(product()->isMergeable()) {
      unsafe_product()->mergeProduct(edp.get());
    } else if(product()->hasIsProductEqual()) {
      if(!product()->isProductEqual(edp.get())) {
        LogError("RunLumiMerging")
              << "ProductHolderBase::mergeTheProduct\n"
              << "Two run/lumi products for the same run/lumi which should be equal are not\n"
              << "Using the first, ignoring the second\n"
              << "className = " << branchDescription().className() << "\n"
              << "moduleLabel = " << moduleLabel() << "\n"
              << "instance = " << productInstanceName() << "\n"
              << "process = " << processName() << "\n";
      }
    } else {
      LogWarning("RunLumiMerging")
          << "ProductHolderBase::mergeTheProduct\n"
          << "Run/lumi product has neither a mergeProduct nor isProductEqual function\n"
          << "Using the first, ignoring the second in merge\n"
          << "className = " << branchDescription().className() << "\n"
          << "moduleLabel = " << moduleLabel() << "\n"
          << "instance = " << productInstanceName() << "\n"
          << "process = " << processName() << "\n";
    }
  }

  bool
  ProductHolderBase::provenanceAvailable() const {
    // If this product is from a the current process,
    // the provenance is available if and only if a product has been put.
    if(branchDescription().produced()) {
      return product() && product()->isPresent();
    }
    // If this product is from a prior process, the provenance is available,
    // although the per event part may have been dropped.
    return true;
  }

  TypeID
  ProductHolderBase::productType() const {
    return TypeID(product()->wrappedTypeInfo());
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
    return &(productData().provenance());
  }

  void
  ProductHolderBase::write(std::ostream& os) const {
    // This is grossly inadequate. It is also not critical for the
    // first pass.
    os << std::string("ProductHolder for product with ID: ")
       << productID();
  }
}
