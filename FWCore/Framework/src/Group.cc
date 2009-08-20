/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include <string>
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "FWCore/Framework/interface/Group.h"
#include "FWCore/Utilities/interface/ReflexTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using Reflex::Type;
using Reflex::TypeTemplate;

namespace edm {


  Group::Group() :
    product_(),
    branchDescription_(),
    pid_(),
    status_(productstatus::uninitialized()),
    prov_(),
    onDemand_(false) {}


  Group::Group(boost::shared_ptr<EDProduct> edp, ConstBranchDescription const& bd,
      ProductID const& pid, std::auto_ptr<ProductProvenance> productProvenance) :
    product_(edp),
    branchDescription_(new ConstBranchDescription(bd)),
    pid_(pid),
    status_(productProvenance->productStatus()),
    prov_(new Provenance(branchDescription(), pid_, boost::shared_ptr<ProductProvenance>(productProvenance.release()))),
    onDemand_(false) {
  }

  Group::Group(boost::shared_ptr<EDProduct> edp, ConstBranchDescription const& bd,
      ProductID const& pid, boost::shared_ptr<ProductProvenance> productProvenance) :
    product_(edp),
    branchDescription_(new ConstBranchDescription(bd)),
    pid_(pid),
    status_(productProvenance->productStatus()),
    prov_(new Provenance(branchDescription(), pid_, productProvenance)),
    onDemand_(false) {
  }

  Group::Group(ConstBranchDescription const& bd,
      ProductID const& pid, boost::shared_ptr<ProductProvenance> productProvenance) :
    product_(),
    branchDescription_(new ConstBranchDescription(bd)),
    pid_(pid),
    status_(productProvenance->productStatus()),
    prov_(new Provenance(branchDescription(), pid_, productProvenance)),
    onDemand_(false) {
  }

  Group::Group(ConstBranchDescription const& bd, ProductID const& pid, ProductStatus const& status) :
    product_(),
    branchDescription_(new ConstBranchDescription(bd)),
    pid_(pid),
    status_(status),
    prov_(),
    onDemand_(productstatus::unscheduledProducerNotRun(status)) {
  }

  Group::Group(ConstBranchDescription const& bd, ProductID const& pid) :
    product_(),
    branchDescription_(new ConstBranchDescription(bd)),
    pid_(pid),
    status_(productstatus::uninitialized()),
    prov_(),
    onDemand_(false) {
  }

  Group::~Group() {
  }

  void 
  Group::setProduct(std::auto_ptr<EDProduct> prod) const {
    assert (!product_);
    product_.reset(prod.release());  // Group takes ownership
    assert (product_);
  }
  void
  Group::updateStatus() const {
    if (product_) {
      if(product_->isPresent()) {
        status_ = productstatus::present();
      } else {
        ProductStatus provStatus = (prov_ && prov_->productProvenancePtr()) ?
				   prov_->productProvenance().productStatus() :
				   productstatus::uninitialized();
        if (productstatus::dropped(provStatus)) {
	  // fixes a backward compatibility problem
	  prov_->productProvenance().setStatus(productstatus::uninitialized());
        }
        if (productstatus::uninitialized(provStatus) || productstatus::unknown(provStatus)) {
	  status_ = productstatus::neverCreated();
        } else {
	  assert(productstatus::notPresent(provStatus));
	  status_ = provStatus;
        }
      }
    } else if (prov_->productProvenancePtr()) {
      status_ = prov_->productProvenance().productStatus();
    } else {
      status_ = productstatus::dropped();
    }
  }
  
  bool
  Group::onDemand() const {
    return onDemand_;
  }

  // This routine returns true if it is known that currently there is no real product.
  // If there is a real product, it returns false.
  // If it is not known if there is a real product, it returns false.
  bool 
  Group::productUnavailable() const { 
    if (onDemand()) return false;
    bool unavailable = productstatus::notPresent(status());
    // If this product is from a the current process,
    // the product is available if and only if a product has been put.
    if (branchDescription().produced()) {
      assert (!productstatus::presenceUnknown(status()));
      assert ((product_ && product_->isPresent()) == !unavailable);
    }
    // The product is from a prior process.
    // if there is a product, we know if it is real or a dummy.
    else if (product_) {
      assert (!productstatus::presenceUnknown(status()));
      assert (product_->isPresent() == !productstatus::notPresent(status()));
    }
    return unavailable;
  }

  bool 
  Group::provenanceAvailable() const { 
    // If this product is from a the current process,
    // the provenance is available if and only if a product has been put.
    if (branchDescription().produced()) {
      return product_ && product_->isPresent();
    }
    // If this product is from a prior process, the provenance is available,
    // although the per event part may have been dropped.
    return true;
  }

  void  
  Group::swap(Group& other) {
    std::swap(product_, other.product_);
    std::swap(pid_, other.pid_);
    std::swap(branchDescription_, other.branchDescription_);
    std::swap(status_, other.status_);
    std::swap(prov_, other.prov_);
    std::swap(onDemand_, other.onDemand_);
  }

  void
  Group::replace(Group& g) {
    this->swap(g);
  }

  Type
  Group::productType() const
  {
    return Type::ByTypeInfo(typeid(*product()));
  }

  bool
  Group::isMatchingSequence(Type const& wantedElementType) const
  {
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

  Provenance const *
  Group::provenance() const {
    if (!prov_.get()) {
      prov_.reset(new Provenance(branchDescription(), pid_));
    }
    return prov_.get();
  }

  void
  Group::write(std::ostream& os) const 
  {
    // This is grossly inadequate. It is also not critical for the
    // first pass.
    os << std::string("Group for product with ID: ")
       << pid_;
  }

  void
  Group::mergeGroup(Group * newGroup) {

    if (status() != newGroup->status()) {
      throw edm::Exception(edm::errors::Unknown, "Merging")
        << "Group::mergeGroup(), Trying to merge two run products or two lumi products.\n"
        << "The products have different creation status's.\n"
        << "For example \"present\" and \"notCreated\"\n"
        << "The Framework does not currently support this and probably\n"
        << "should not support this case.\n"
        << "Likely a problem exists in the producer that created these\n"
        << "products.  If this problem cannot be reasonably resolved by\n"
        << "fixing the producer module, then contact the Framework development\n"
        << "group with details so we can discuss whether and how to support this\n"
        << "use case.\n"
        << "className = " << branchDescription().className() << "\n"
        << "moduleLabel = " << moduleLabel() << "\n"
        << "instance = " << productInstanceName() << "\n"
        << "process = " << processName() << "\n";
    }

    if (!provenance()->productProvenanceSharedPtr()) {
      return;
    }

    if (!productUnavailable() && !newGroup->productUnavailable()) {

      if (product_->isMergeable()) {
        product_->mergeProduct(newGroup->product_.get());
      }
      else if (product_->hasIsProductEqual()) {

        if (!product_->isProductEqual(newGroup->product_.get())) {
          edm::LogWarning  ("RunLumiMerging") 
            << "Group::mergeGroup\n" 
            << "Two run/lumi products for the same run/lumi which should be equal are not\n"
            << "Using the first, ignoring the second\n"
            << "className = " << branchDescription().className() << "\n"
            << "moduleLabel = " << moduleLabel() << "\n"
            << "instance = " << productInstanceName() << "\n"
            << "process = " << processName() << "\n";
        }
      }
      else {
        edm::LogWarning  ("RunLumiMerging") 
          << "Group::mergeGroup\n" 
          << "Run/lumi product has neither a mergeProduct nor isProductEqual function\n"
          << "Using the first, ignoring the second in merge\n"
          << "className = " << branchDescription().className() << "\n"
          << "moduleLabel = " << moduleLabel() << "\n"
          << "instance = " << productInstanceName() << "\n"
          << "process = " << processName() << "\n";
      }
    }
  }
}
