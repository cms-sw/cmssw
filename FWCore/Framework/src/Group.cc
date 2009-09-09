/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include <string>
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
    assert(branchDescription().produced());
    assert(edp.get() != 0);
    assert(!product());
    assert(!productProvenancePtr());
    assert(status() != Present);
    assert(status() != Uninitialized);
    setProvenance(productProvenance);
    assert(productProvenancePtr());
    setProduct(edp);
    status_() = Present;
  }

  void
  ProducedGroup::putOrMergeProduct_(
	std::auto_ptr<EDProduct> edp,
	boost::shared_ptr<ProductProvenance> productProvenance) {
    if (productUnavailable()) {
      putProduct_(edp, productProvenance);
    } else {
      assert(productProvenancePtr());
      assert(status() == Present);
      assert (productProvenancePtr()->productStatus() == productProvenance->productStatus());
      productProvenancePtr() = productProvenance;
      mergeProduct(edp);
    }
  }

  void
  ProducedGroup::putOrMergeProduct_(std::auto_ptr<EDProduct> edp) const {
    assert(0);
  }

  void
  ProducedGroup::putProduct_(std::auto_ptr<EDProduct> edp) const {
    assert(0);
  }

  void
  ProducedGroup::resolveProvenance_(boost::shared_ptr<BranchMapper>) const {
  }

  void
  InputGroup::putProduct_(
	std::auto_ptr<EDProduct> edp,
	boost::shared_ptr<ProductProvenance> productProvenance) {
    assert(!product());
    assert(!productProvenancePtr());
    assert(status() != Present);
    setProvenance(productProvenance);
    assert(productProvenancePtr());
    setProduct(edp);
    updateStatus();
  }

  void
  InputGroup::putOrMergeProduct_(
	std::auto_ptr<EDProduct> edp,
	boost::shared_ptr<ProductProvenance> productProvenance) {
    assert(0);
  }

  void
  InputGroup::putOrMergeProduct_(std::auto_ptr<EDProduct> edp) const {
    if (!product()) {
      setProduct(edp);
    } else {
      mergeProduct(edp);
    }
    updateStatus();
  }

  void
  InputGroup::putProduct_(std::auto_ptr<EDProduct> edp) const {
    assert(!product());
    setProduct(edp);
    updateStatus();
  }

  void
  InputGroup::resolveProvenance_(boost::shared_ptr<BranchMapper> store) const {
    if (!productProvenancePtr()) {
      provenance()->setStore(store);
      provenance()->resolve();
      updateStatus();
    }
  }

  void
  Group::mergeProduct(std::auto_ptr<EDProduct> edp) const {
  if (product()->isMergeable()) {
    product()->mergeProduct(edp.get());
  } else if (product()->hasIsProductEqual()) {
    if (!product()->isProductEqual(edp.get())) {
      LogWarning("RunLumiMerging")
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
    if(TypeID(prod).className() != branchDescription_->wrappedName()) {
	std::string name(TypeID(prod).className());
	throw edm::Exception(errors::EventCorruption)
	  << "Product on branch " << branchDescription_->branchName() << " is of wrong type.\n"
	  << "It is supposed to be of type " << branchDescription_->wrappedName() << ".\n"
	  << "It is actually of type " << name << ".\n";
    }
  }

  void
  Group::setProduct(std::auto_ptr<EDProduct> prod) const {
    assert (!product());
    groupData().product_.reset(prod.release());  // Group takes ownership
  }

  void
  Group::setProvenance(boost::shared_ptr<ProductProvenance> prov) const {
    groupData().prov_->setProductProvenance(prov);
  }

  void
  InputGroup::setStatus(ProductStatus const& status) const {
    theStatus_ = static_cast<GroupStatus>(status);
  }

  void
  InputGroup::updateStatus() const {
    if (product()) {
      if(product()->isPresent()) {
        theStatus_ = Present;
      } else {
        ProductStatus provStatus = (provenance() && provenance()->productProvenancePtr()) ?
				   provenance()->productProvenance().productStatus() :
				   productstatus::uninitialized();
        if (productstatus::dropped(provStatus)) {
	  // fixes a backward compatibility problem
	  provenance()->productProvenance().setStatus(productstatus::uninitialized());
        }
        if (productstatus::uninitialized(provStatus) || productstatus::unknown(provStatus)) {
	  theStatus_ = NeverCreated;
        } else {
	  assert(productstatus::notPresent(provStatus));
	  setStatus(provStatus);
        }
      }
    } else if (provenance()->productProvenancePtr()) {
      setStatus(provenance()->productProvenance().productStatus());
    }
  }

  // This routine returns true if it is known that currently there is no real product.
  // If there is a real product, it returns false.
  // If it is not known if there is a real product, it returns false.
  bool
  InputGroup::productUnavailable_() const {
    // If there is a product, we know if it is real or a dummy.
    if (product()) {
      bool unavailable = !(product()->isPresent());
      assert (!productstatus::presenceUnknown(status()));
      assert(unavailable == productstatus::notPresent(status()));
      return unavailable;
    }
    return productstatus::notPresent(status());
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

    return
      is_sequence
      ? (elementType==wantedElementType ||
	 elementType.HasBase(wantedElementType))
      : false;
  }

  Provenance*
  Group::provenance() const {
    if (!groupData().prov_.get()) {
      groupData().prov_.reset(new Provenance(branchDescription(), productID()));
    }
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
