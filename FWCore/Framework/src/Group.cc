/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "FWCore/Framework/interface/Group.h"
#include "FWCore/Utilities/interface/ReflexTools.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using Reflex::Type;
using Reflex::TypeTemplate;

namespace edm {

  Group::Group() {}

  Group::~Group() {}
  InputGroup::~InputGroup() {}
  ProducedGroup::~ProducedGroup() {}
  ScheduledGroup::~ScheduledGroup() {}
  UnscheduledGroup::~UnscheduledGroup() {}
  SourceGroup::~SourceGroup() {}

  void
  ProducedGroup::putProduct_(
        std::auto_ptr<EDProduct> edp,
        boost::shared_ptr<ProductProvenance> productProvenance) {
    if (product()) {
      throw edm::Exception(errors::InsertFailure)
          << "Attempt to insert more than one product on branch " << groupData().branchDescription_->branchName() << "\n";
    }
    assert(branchDescription().produced());
    assert(edp.get() != 0);
    assert(!provenance()->productProvenanceResolved());
    assert(status() != Present);
    assert(status() != Uninitialized);
    setProductProvenance(productProvenance);
    assert(provenance()->productProvenanceResolved());
    groupData().product_.reset(edp.release());  // Group takes ownership
    status_() = Present;
  }

  void
  ProducedGroup::mergeProduct_(
        std::auto_ptr<EDProduct> edp,
        boost::shared_ptr<ProductProvenance> productProvenance) {
    assert(provenance()->productProvenanceResolved());
    assert(status() == Present);
    assert (productProvenancePtr()->productStatus() == productProvenance->productStatus());
    productProvenancePtr() = productProvenance;
    mergeTheProduct(edp);
  }

  bool
  ProducedGroup::putOrMergeProduct_() const {
    return productUnavailable();
  }

  void
  ProducedGroup::mergeProduct_(std::auto_ptr<EDProduct> edp) const {
    assert(0);
  }

  void
  ProducedGroup::putProduct_(std::auto_ptr<EDProduct> edp) const {
    assert(0);
  }

  void
  InputGroup::putProduct_(
        std::auto_ptr<EDProduct> edp,
        boost::shared_ptr<ProductProvenance> productProvenance) {
    assert(!product());
    assert(!provenance()->productProvenanceResolved());
    setProductProvenance(productProvenance);
    assert(provenance()->productProvenanceResolved());
    setProduct(edp);
  }

  void
  InputGroup::mergeProduct_(
        std::auto_ptr<EDProduct> edp,
        boost::shared_ptr<ProductProvenance> productProvenance) {
    assert(0);
  }

  void
  InputGroup::mergeProduct_(std::auto_ptr<EDProduct> edp) const {
    mergeTheProduct(edp);
  }

  bool
  InputGroup::putOrMergeProduct_() const {
    return (!product());
  }

  void
  InputGroup::putProduct_(std::auto_ptr<EDProduct> edp) const {
    assert(!product());
    setProduct(edp);
  }

  void
  Group::mergeTheProduct(std::auto_ptr<EDProduct> edp) const {
    if (product()->isMergeable()) {
      product()->mergeProduct(edp.get());
    } else if (product()->hasIsProductEqual()) {
      if (!product()->isProductEqual(edp.get())) {
        LogError("RunLumiMerging")
              << "Group::mergeGroup\n"
              << "Two run/lumi products for the same run/lumi which should be equal are not\n"
              << "Using the first, ignoring the second\n"
              << "className = " << branchDescription().className() << "\n"
              << "moduleLabel = " << moduleLabel() << "\n"
              << "instance = " << productInstanceName() << "\n"
              << "process = " << processName() << "\n";
      }
    } else {
      LogWarning("RunLumiMerging")
          << "Group::mergeGroup\n"
          << "Run/lumi product has neither a mergeProduct nor isProductEqual function\n"
          << "Using the first, ignoring the second in merge\n"
          << "className = " << branchDescription().className() << "\n"
          << "moduleLabel = " << moduleLabel() << "\n"
          << "instance = " << productInstanceName() << "\n"
          << "process = " << processName() << "\n";
    }
  }

  void
  GroupData::checkType(EDProduct const& prod) const {
    // Check if the types match.
    TypeID typeID(prod.dynamicTypeInfo());
    if (typeID != branchDescription_->typeID()) {
      // Types do not match.
      throw edm::Exception(errors::EventCorruption)
          << "Product on branch " << branchDescription_->branchName() << " is of wrong type.\n"
          << "It is supposed to be of type " << branchDescription_->className() << ".\n"
          << "It is actually of type " << typeID.className() << ".\n";
    }
  }

  void
  InputGroup::setProduct(std::auto_ptr<EDProduct> prod) const {
    assert (!product());
    if (prod.get() == 0 || !prod->isPresent()) {
      setProductUnavailable();
    }
    groupData().product_.reset(prod.release());  // Group takes ownership
  }

  void
  Group::setProductProvenance(boost::shared_ptr<ProductProvenance> prov) const {
    groupData().prov_->setProductProvenance(prov);
  }

  // This routine returns true if it is known that currently there is no real product.
  // If there is a real product, it returns false.
  // If it is not known if there is a real product, it returns false.
  bool
  InputGroup::productUnavailable_() const {
    if (productIsUnavailable()) {
      return true;
    }
    // If there is a product, we know if it is real or a dummy.
    if (product()) {
      bool unavailable = !(product()->isPresent());
      if (unavailable) {
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
  ProducedGroup::productUnavailable_() const {
    // If unscheduled production, the product is potentially available.
    if (onDemand()) return false;
    // The product is available if and only if a product has been put.
    bool unavailable = !(product() && product()->isPresent());
    assert (!productstatus::presenceUnknown(status()));
    assert(unavailable == productstatus::notPresent(status()));
    return unavailable;
  }

  bool
  Group::provenanceAvailable() const {
    // If this product is from a the current process,
    // the provenance is available if and only if a product has been put.
    if (branchDescription().produced()) {
      return product() && product()->isPresent();
    }
    // If this product is from a prior process, the provenance is available,
    // although the per event part may have been dropped.
    return true;
  }

  Type
  Group::productType() const {
    return Type::ByTypeInfo(typeid(*product()));
  }

  bool
  Group::isMatchingSequence(Type const& wantedElementType) const {
    Type value_type;
    bool is_sequence = is_sequence_wrapper(productType(), value_type);

    // If our product is not a sequence, we can't match...
    if (!is_sequence) return false;

    Type elementType = value_type; // this is not true for RefVector...

    TypeTemplate valueTypeTemplate = value_type.TemplateFamily();

    return (elementType==wantedElementType ||
            elementType.HasBase(wantedElementType));
  }

  void
  Group::setProvenance(boost::shared_ptr<BranchMapper> mapper, ProductID const& pid) {
    assert(!groupData().prov_);
    groupData().prov_.reset(new Provenance(branchDescription(), pid));
    groupData().prov_->setStore(mapper);
  }

  void
  Group::setProvenance(boost::shared_ptr<BranchMapper> mapper) {
    if (!groupData().prov_) {
      groupData().prov_.reset(new Provenance(branchDescription(), ProductID()));
    }
    groupData().prov_->setStore(mapper);
  }

  Provenance*
  Group::provenance() const {
    return groupData().prov_.get();
  }

  void
  Group::write(std::ostream& os) const {
    // This is grossly inadequate. It is also not critical for the
    // first pass.
    os << std::string("Group for product with ID: ")
       << productID();
  }
}
