/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/Group.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <cassert>

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
        WrapperHolder const& edp,
        boost::shared_ptr<ProductProvenance> productProvenance) {
    if(product()) {
      throw Exception(errors::InsertFailure)
          << "Attempt to insert more than one product on branch " << productData().branchDescription()->branchName() << "\n";
    }
    assert(branchDescription().produced());
    assert(edp.isValid());
    assert(!provenance()->productProvenanceResolved());
    assert(status() != Present);
    assert(status() != Uninitialized);
    setProductProvenance(productProvenance);
    assert(provenance()->productProvenanceResolved());
    if(productData().getInterface() != 0) {
      assert(productData().getInterface() == edp.interface());
    }
    productData().wrapper_ = edp.product();
    status_() = Present;
  }

  void
  ProducedGroup::mergeProduct_(
        WrapperHolder const& edp,
        boost::shared_ptr<ProductProvenance> productProvenance) {
    assert(provenance()->productProvenanceResolved());
    assert(status() == Present);
    productProvenancePtr() = productProvenance;
    mergeTheProduct(edp);
  }

  bool
  ProducedGroup::putOrMergeProduct_() const {
    return productUnavailable();
  }

  void
  ProducedGroup::mergeProduct_(WrapperHolder const&) const {
    assert(0);
  }

  void
  ProducedGroup::putProduct_(WrapperHolder const&) const {
    assert(0);
  }

  void
  InputGroup::putProduct_(
        WrapperHolder const& edp,
        boost::shared_ptr<ProductProvenance> productProvenance) {
    assert(!product());
    assert(!provenance()->productProvenanceResolved());
    setProductProvenance(productProvenance);
    assert(provenance()->productProvenanceResolved());
    setProduct(edp);
  }

  void
  InputGroup::mergeProduct_(
        WrapperHolder const&,
        boost::shared_ptr<ProductProvenance>) {
    assert(0);
  }

  void
  InputGroup::mergeProduct_(WrapperHolder const& edp) const {
    mergeTheProduct(edp);
  }

  bool
  InputGroup::putOrMergeProduct_() const {
    return(!product());
  }

  void
  InputGroup::putProduct_(WrapperHolder const& edp) const {
    assert(!product());
    setProduct(edp);
  }

  void
  Group::mergeTheProduct(WrapperHolder const& edp) const {
    if(wrapper().isMergeable()) {
      wrapper().mergeProduct(edp.wrapper());
    } else if(wrapper().hasIsProductEqual()) {
      if(!wrapper().isProductEqual(edp.wrapper())) {
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
  InputGroup::setProduct(WrapperHolder const& prod) const {
    assert (!product());
    if(!prod.isValid() || !prod.isPresent()) {
      setProductUnavailable();
    }
    assert(productData().getInterface() == prod.interface() || !prod.isValid());
    productData().wrapper_ = prod.product();
  }

  void
  Group::setProductProvenance(boost::shared_ptr<ProductProvenance> prov) const {
    productData().prov_.setProductProvenance(prov);
  }

  // This routine returns true if it is known that currently there is no real product.
  // If there is a real product, it returns false.
  // If it is not known if there is a real product, it returns false.
  bool
  InputGroup::productUnavailable_() const {
    if(productIsUnavailable()) {
      return true;
    }
    // If there is a product, we know if it is real or a dummy.
    if(product()) {
      bool unavailable = !(wrapper().isPresent());
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
  ProducedGroup::productUnavailable_() const {
    // If unscheduled production, the product is potentially available.
    if(onDemand()) return false;
    // The product is available if and only if a product has been put.
    bool unavailable = !(product() && wrapper().isPresent());
    return unavailable;
  }

  bool
  Group::provenanceAvailable() const {
    // If this product is from a the current process,
    // the provenance is available if and only if a product has been put.
    if(branchDescription().produced()) {
      return product() && wrapper().isPresent();
    }
    // If this product is from a prior process, the provenance is available,
    // although the per event part may have been dropped.
    return true;
  }

  Type
  Group::productType() const {
    return Type::ByTypeInfo(wrapper().interface()->wrappedTypeInfo());
  }

  void
  Group::reallyCheckType(WrapperHolder const& prod) const {
    // Check if the types match.
    TypeID typeID(prod.dynamicTypeInfo());
    if(typeID != branchDescription().typeID()) {
      // Types do not match.
      throw Exception(errors::EventCorruption)
          << "Product on branch " << branchDescription().branchName() << " is of wrong type.\n"
          << "It is supposed to be of type " << branchDescription().className() << ".\n"
          << "It is actually of type " << typeID.className() << ".\n";
    }
  }

  void
  Group::setProvenance(boost::shared_ptr<BranchMapper> mapper, ProductID const& pid) {
    //assert(!productData().prov_);
    productData().prov_.setProductID(pid);
    productData().prov_.setStore(mapper);
  }

  void
  Group::setProvenance(boost::shared_ptr<BranchMapper> mapper) {
    productData().prov_.setStore(mapper);
  }

  Provenance*
  Group::provenance() const {
    return &(productData().prov_);
  }

  void
  Group::write(std::ostream& os) const {
    // This is grossly inadequate. It is also not critical for the
    // first pass.
    os << std::string("Group for product with ID: ")
       << productID();
  }
}
