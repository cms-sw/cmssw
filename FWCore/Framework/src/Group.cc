/*----------------------------------------------------------------------
$Id: Group.cc,v 1.33 2008/02/02 21:27:34 wmtan Exp $
----------------------------------------------------------------------*/
#include <string>
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "FWCore/Framework/src/Group.h"
#include "FWCore/Utilities/interface/ReflexTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using ROOT::Reflex::Type;
using ROOT::Reflex::TypeTemplate;

namespace edm {

  Group::Group() :
    product_(),
    provenance_(),
    status_(productstatus::neverCreated()),
    dropped_(false) {}


  Group::Group(std::auto_ptr<Provenance> prov) :
    product_(),
    provenance_(prov.release()),
    status_(productstatus::neverCreated()),
    dropped_(!provenance_->product().present()) {
  }

  Group::Group(ConstBranchDescription const& bd, ProductStatus status) :
    product_(),
    provenance_(new Provenance(bd)),
    status_(status),
    dropped_(!bd.present()) {
  }

  Group::Group(std::auto_ptr<EDProduct> edp,
	       std::auto_ptr<Provenance> prov) :
    product_(edp.release()),
    provenance_(prov.release()),
    status_(productstatus::present()),
    dropped_(false) {
  }

  Group::~Group() {
  }

  ProductStatus
  Group::status() const {
    // for backward compatibility
    if (productstatus::unknown(status_) && product_) {
      status_ = (product_->isPresent() ? productstatus::present() : productstatus::neverCreated());
    }
    return status_;
  }

  bool
  Group::onDemand() const {
    return productstatus::onDemand(status_);
  }

  bool 
  Group::productUnavailable() const { 
    if (onDemand()) return false;
    if (dropped_) return true;
    if (productstatus::unknown(status())) return false;
    return not productstatus::present(status_);

  }

  bool 
  Group::provenanceAvailable() const { 
    if (onDemand()) return false;
    if (productstatus::unknown(status())) return true;
    return productstatus::present(status_);
  }

  void 
  Group::setProduct(std::auto_ptr<EDProduct> prod) const {
    assert (product() == 0);
    product_ = boost::shared_ptr<EDProduct>(prod.release());  // Group takes ownership
  }
  
  void 
  Group::setProvenance(std::auto_ptr<EntryDescription> prov) const {
    assert (entryDescription() == 0);
    provenance_->setEvent(boost::shared_ptr<EntryDescription>(prov.release()));  // Group takes ownership
  }

  void  
  Group::swap(Group& other) {
    std::swap(product_, other.product_);
    std::swap(provenance_, other.provenance_);
    std::swap(status_, other.status_);
    std::swap(dropped_, other.dropped_);
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

  BasicHandle
  Group::makeBasicHandle() const
  {
    return BasicHandle(product_.get(), provenance_.get());
  }

  void
  Group::write(std::ostream& os) const 
  {
    // This is grossly inadequate. It is also not critical for the
    // first pass.
    os << std::string("Group for product with ID: ")
       << provenance_->productID();
  }

  void
  Group::mergeGroup(Group * newGroup) {

    if (productUnavailable() && !newGroup->productUnavailable()) {
      provenance_->entryDescription()->mergeEntryDescription(newGroup->entryDescription());
      status_ = productstatus::present();
      dropped_ = false;
      std::swap(product_, newGroup->product_);        
    }
    else if (!productUnavailable() && !newGroup->productUnavailable()) {

      provenance_->entryDescription()->mergeEntryDescription(newGroup->entryDescription());

      if (product_->isMergeable()) {
        product_->mergeProduct(newGroup->product_.get());
      }
      else if (product_->hasIsProductEqual()) {

        if (!product_->isProductEqual(newGroup->product_.get())) {
          edm::LogWarning  ("RunLumiMerging") 
            << "Group::mergeGroup\n" 
            << "Two run/lumi products for the same run/lumi which should be equal are not\n"
            << "Using the first, ignoring the second\n"
            << "className = " << provenance().className() << "\n"
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
          << "className = " << provenance().className() << "\n"
          << "moduleLabel = " << moduleLabel() << "\n"
          << "instance = " << productInstanceName() << "\n"
          << "process = " << processName() << "\n";
      }
    }
    else {
      provenance_->entryDescription()->mergeEntryDescription(newGroup->entryDescription());
    }
  }
}
