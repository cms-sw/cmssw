/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/ProductHolder.h"
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
  InputProductHolder::~InputProductHolder() {}
  ProducedProductHolder::~ProducedProductHolder() {}
  ScheduledProductHolder::~ScheduledProductHolder() {}
  UnscheduledProductHolder::~UnscheduledProductHolder() {}
  SourceProductHolder::~SourceProductHolder() {}
  AliasProductHolder::~AliasProductHolder() {}
  NoProcessProductHolder::~NoProcessProductHolder() {}

  ProductData const*
  InputProductHolder::resolveProduct_(ResolveStatus& resolveStatus, bool,
                                      ModuleCallingContext const* mcc) const {
    if(productWasDeleted()) {
      throwProductDeletedException();
    }
    if(!productUnavailable()) {
      principal_->readFromSource(*this, mcc);
      // If the product is a dummy filler, product holder will now be marked unavailable.
      if(product() && !productUnavailable()) {
        // Found the match
        resolveStatus = ProductFound;
        return &productData_;
      }
    }
    resolveStatus = ProductNotFound;
    return nullptr;
  }

  ProductData const*
  ScheduledProductHolder::resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess,
                                          ModuleCallingContext const*) const {
    if (!skipCurrentProcess) {
      if(productWasDeleted()) {
        throwProductDeletedException();
      }
      if(product() && product()->isPresent()) {
        resolveStatus = ProductFound;
        return &productData_;
      }
    }
    resolveStatus = ProductNotFound;
    return nullptr;
  }

  ProductData const*
  SourceProductHolder::resolveProduct_(ResolveStatus& resolveStatus, bool skipCurrentProcess,
                                       ModuleCallingContext const*) const {
    if (!skipCurrentProcess) {
      if(productWasDeleted()) {
        throwProductDeletedException();
      }
      if(product() && product()->isPresent()) {
        resolveStatus = ProductFound;
        return &productData_;
      }
    }
    resolveStatus = ProductNotFound;
    return nullptr;
  }

  ProductData const*
  UnscheduledProductHolder::resolveProduct_(ResolveStatus& resolveStatus,
                                            bool skipCurrentProcess,
                                            ModuleCallingContext const* mcc) const {
    if (!skipCurrentProcess) {
      if(productWasDeleted()) {
        throwProductDeletedException();
      }
      if(product() && product()->isPresent()) {
        resolveStatus = ProductFound;
        return &productData_;
      }
      principal_->unscheduledFill(moduleLabel(), mcc);
      if(product() && product()->isPresent()) {
        resolveStatus = ProductFound;
        return &productData_;
      }
    }
    resolveStatus = ProductNotFound;
    return nullptr;
  }

  void
  ProducedProductHolder::putProduct_(
        std::unique_ptr<WrapperBase> edp,
        ProductProvenance const& productProvenance) {
    if(product()) {
      throw Exception(errors::InsertFailure)
          << "Attempt to insert more than one product on branch " << branchDescription().branchName() << "\n";
    }
    assert(branchDescription().produced());
    assert(edp.get() != nullptr);
    assert(!provenance()->productProvenanceValid());
    assert(status() != Present);
    assert(status() != Uninitialized);
    setProductProvenance(productProvenance);
    assert(provenance()->productProvenanceValid());
    productData().wrapper_ = std::move(edp); // ProductHolder takes ownership
    status_() = Present;
  }

  void
  ProducedProductHolder::mergeProduct_(
        std::unique_ptr<WrapperBase> edp,
        ProductProvenance& productProvenance) {
    assert(provenance()->productProvenanceValid());
    assert(status() == Present);
    setProductProvenance(productProvenance);
    mergeTheProduct(std::move(edp));
  }

  bool
  ProducedProductHolder::putOrMergeProduct_() const {
    return productUnavailable();
  }

  void
  ProducedProductHolder::mergeProduct_(std::unique_ptr<WrapperBase> edp) const {
    assert(status() == Present);
    mergeTheProduct(std::move(edp));
  }

  void
  ProducedProductHolder::putProduct_(std::unique_ptr<WrapperBase> edp) const {
    if(product()) {
      throw Exception(errors::InsertFailure)
          << "Attempt to insert more than one product on branch " << branchDescription().branchName() << "\n";
    }
    assert(branchDescription().produced());
    assert(edp.get() != nullptr);
    assert(status() != Present);
    assert(status() != Uninitialized);
    productData().wrapper_ = std::move(edp);  // ProductHolder takes ownership
    status_() = Present;
  }

  void
  InputProductHolder::putProduct_(
        std::unique_ptr<WrapperBase> edp,
        ProductProvenance const& productProvenance) {
    assert(!product());
    assert(!provenance()->productProvenanceValid());
    setProductProvenance(productProvenance);
    assert(provenance()->productProvenanceValid());
    setProduct(std::move(edp));
  }

  void
  InputProductHolder::mergeProduct_(
        std::unique_ptr<WrapperBase>,
        ProductProvenance&) {
    assert(nullptr);
  }

  void
  InputProductHolder::mergeProduct_(std::unique_ptr<WrapperBase> edp) const {
    mergeTheProduct(std::move(edp));
  }

  bool
  InputProductHolder::putOrMergeProduct_() const {
    return(!product());
  }

  void
  InputProductHolder::putProduct_(std::unique_ptr<WrapperBase> edp) const {
    assert(!product());
    setProduct(std::move(edp));
  }

  void
  ProductHolderBase::mergeTheProduct(std::unique_ptr<WrapperBase> edp) const {
    if(product()->isMergeable()) {
      product()->mergeProduct(edp.get());
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

  void
  InputProductHolder::setProduct(std::unique_ptr<WrapperBase> prod) const {
    assert (!product());
    if(prod.get() == nullptr || !prod->isPresent()) {
      setProductUnavailable();
    }
    productData().wrapper_ = std::move(prod);  // ProductHolder takes ownership
  }

  void InputProductHolder::setProvenance_(std::shared_ptr<ProductProvenanceRetriever> provRetriever, ProcessHistory const& ph, ProductID const& pid) {
    productData().prov_.setProductID(pid);
    productData().prov_.setStore(provRetriever);
    productData().prov_.setProcessHistory(ph);
  }

  void InputProductHolder::setProcessHistory_(ProcessHistory const& ph) {
    productData().prov_.setProcessHistory(ph);
  }

  ProductProvenance* InputProductHolder::productProvenancePtr_() const {
    return provenance()->productProvenance();
  }

  void InputProductHolder::resetProductData_() {
    productData().resetProductData();
    resetStatus();
  }

  bool InputProductHolder::singleProduct_() const {
    return true;
  }

  void InputProductHolder::setPrincipal_(Principal* principal) {
    principal_ = principal;
  }

  void
  ProductHolderBase::setProductProvenance(ProductProvenance const& prov) const {
    productData().prov_.setProductProvenance(prov);
  }

  // This routine returns true if it is known that currently there is no real product.
  // If there is a real product, it returns false.
  // If it is not known if there is a real product, it returns false.
  bool
  InputProductHolder::productUnavailable_() const {
    if(productIsUnavailable()) {
      return true;
    }
    // If there is a product, we know if it is real or a dummy.
    if(product()) {
      bool unavailable = !(product()->isPresent());
      if(unavailable) {
        setProductUnavailable();
      }
      return unavailable;
    }
    return false;
  }

  // This routine returns true if it is known that currently there is no real product.
  // If there is a real product, it returns false.
  // If it is not known if there is a real product, it returns false.
  bool
  ProducedProductHolder::productUnavailable_() const {
    // If unscheduled production, the product is potentially available.
    if(onDemand()) return false;
    // The product is available if and only if a product has been put.
    bool unavailable = !(product() && product()->isPresent());
    return unavailable;
  }

  // This routine returns true if the product was deleted early in order to save memory
  bool
  ProducedProductHolder::productWasDeleted_() const {
    return status() == ProductDeleted;
  }

  void 
  ProducedProductHolder::setProductDeleted_() {
    status() = ProductDeleted;
  }

  void ProducedProductHolder::setProvenance_(std::shared_ptr<ProductProvenanceRetriever> provRetriever, ProcessHistory const& ph, ProductID const& pid) {
    productData().prov_.setProductID(pid);
    productData().prov_.setStore(provRetriever);
    productData().prov_.setProcessHistory(ph);
  }

  void ProducedProductHolder::setProcessHistory_(ProcessHistory const& ph) {
    productData().prov_.setProcessHistory(ph);
  }
  
  ProductProvenance* ProducedProductHolder::productProvenancePtr_() const {
    return provenance()->productProvenance();
  }

  void ProducedProductHolder::resetProductData_() {
    productData().resetProductData();
    resetStatus();
  }

  bool ProducedProductHolder::singleProduct_() const {
    return true;
  }

  void ProducedProductHolder::setPrincipal_(Principal*) {
    throw Exception(errors::LogicError)
      << "ProducedProductHolder::setPrincipal__() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
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

  Provenance*
  ProductHolderBase::provenance() const {
    return &(productData().prov_);
  }

  void
  ProductHolderBase::write(std::ostream& os) const {
    // This is grossly inadequate. It is also not critical for the
    // first pass.
    os << std::string("ProductHolder for product with ID: ")
       << productID();
  }

  NoProcessProductHolder::
  NoProcessProductHolder(std::vector<ProductHolderIndex> const&  matchingHolders,
                         std::vector<bool> const& ambiguous,
                         Principal* principal) : 
    matchingHolders_(matchingHolders),
    ambiguous_(ambiguous),
    principal_(principal) {
    assert(ambiguous_.size() == matchingHolders_.size());
  }

  ProductData const& NoProcessProductHolder::getProductData() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::getProductData() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  ProductData& NoProcessProductHolder::getProductData() {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::getProductData() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  ProductData const* NoProcessProductHolder::resolveProduct_(ResolveStatus& resolveStatus,
                                                             bool skipCurrentProcess,
                                                             ModuleCallingContext const* mcc) const {
    std::vector<unsigned int> const& lookupProcessOrder = principal_->lookupProcessOrder();
    for(unsigned int k : lookupProcessOrder) {
      assert(k < ambiguous_.size());
      if(k == 0) break; // Done
      if(ambiguous_[k]) {
        resolveStatus = Ambiguous;
        return nullptr;
      }
      if (matchingHolders_[k] != ProductHolderIndexInvalid) {
        ProductHolderBase const* productHolder = principal_->getProductHolderByIndex(matchingHolders_[k]);
        ProductData const* pd =  productHolder->resolveProduct(resolveStatus, skipCurrentProcess, mcc);
        if(pd != nullptr) return pd;
      }
    }
    resolveStatus = ProductNotFound;
    return nullptr;
  }

  void AliasProductHolder::setProvenance_(std::shared_ptr<ProductProvenanceRetriever> provRetriever, ProcessHistory const& ph, ProductID const& pid) {
    productData().prov_.setProductID(pid);
    productData().prov_.setStore(provRetriever);
    productData().prov_.setProcessHistory(ph);
  }

  void AliasProductHolder::setProcessHistory_(ProcessHistory const& ph) {
    productData().prov_.setProcessHistory(ph);
  }

  ProductProvenance* AliasProductHolder::productProvenancePtr_() const {
    return provenance()->productProvenance();
  }

  void AliasProductHolder::resetProductData_() {
    productData().resetProductData();
    resetStatus();
  }

  bool AliasProductHolder::singleProduct_() const {
    return true;
  }

  void AliasProductHolder::setPrincipal_(Principal*) {
  }

  void NoProcessProductHolder::swap_(ProductHolderBase& rhs) {
    NoProcessProductHolder& other = dynamic_cast<NoProcessProductHolder&>(rhs);
    ambiguous_.swap(other.ambiguous_);
    matchingHolders_.swap(other.matchingHolders_);
    std::swap(principal_, other.principal_);
  }

  void NoProcessProductHolder::resetStatus_() {
  }

  void NoProcessProductHolder::setProvenance_(std::shared_ptr<ProductProvenanceRetriever> , ProcessHistory const& , ProductID const& ) {
  }

  void NoProcessProductHolder::setProcessHistory_(ProcessHistory const& ) {
  }

  ProductProvenance* NoProcessProductHolder::productProvenancePtr_() const {
    return nullptr;
  }

  void NoProcessProductHolder::resetProductData_() {
  }

  bool NoProcessProductHolder::singleProduct_() const {
    return false;
  }

  void NoProcessProductHolder::setPrincipal_(Principal* ) {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::setPrincipal__() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  bool NoProcessProductHolder::onDemand_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::onDemand_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  bool NoProcessProductHolder::productUnavailable_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::productUnavailable_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  bool NoProcessProductHolder::productWasDeleted_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::productWasDeleted_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductHolder::putProduct_(std::unique_ptr<WrapperBase> , ProductProvenance const& ) {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::putProduct_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductHolder::putProduct_(std::unique_ptr<WrapperBase> ) const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::putProduct_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductHolder::mergeProduct_(std::unique_ptr<WrapperBase> , ProductProvenance& ) {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::mergeProduct_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductHolder::mergeProduct_(std::unique_ptr<WrapperBase>) const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::mergeProduct_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  bool NoProcessProductHolder::putOrMergeProduct_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::putOrMergeProduct_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductHolder::checkType_(WrapperBase const&) const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::checkType_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductHolder::setProductDeleted_() {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::setProductDeleted_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  BranchDescription const& NoProcessProductHolder::branchDescription_() const {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::branchDescription_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }

  void NoProcessProductHolder::resetBranchDescription_(std::shared_ptr<BranchDescription const>) {
    throw Exception(errors::LogicError)
      << "NoProcessProductHolder::resetBranchDescription_() not implemented and should never be called.\n"
      << "Contact a Framework developer\n";
  }
}
