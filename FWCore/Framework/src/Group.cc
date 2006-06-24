/*----------------------------------------------------------------------
$Id: Group.cc,v 1.10 2006/03/05 21:45:07 chrjones Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/Group.h"

namespace edm
{
  Group::Group(std::auto_ptr<Provenance> prov) :
    product_(),
    provenance_(prov.release()),
    accessible_(true) {
  }

  Group::Group(std::auto_ptr<EDProduct> edp,
	       std::auto_ptr<Provenance> prov,
	       bool acc) :
    product_(edp.release()),
    provenance_(prov.release()),
    accessible_(acc) {
  }

  Group::~Group() {
    delete product_;
    delete provenance_;
  }

  bool 
  Group::isAccessible() const { 
      return 
	accessible_ and
	(provenance_->creatorStatus() == BranchEntryDescription::Success);
  }

  void 
  Group::setID(ProductID const& id) {
      provenance_->event.productID_ = id;
  }

  void 
  Group::setProduct(std::auto_ptr<EDProduct> prod) const {
    assert (product() == 0);
    product_ = prod.release();  // Group takes ownership
  }
  
  void  
  Group::swap(Group& other) {
    std::swap(accessible_, other.accessible_);
    std::swap(product_,other.product_);
    std::swap(provenance_,other.provenance_);
  }
   void  
   Group::swapProduct(Group& other) {
      std::swap(product_,other.product_);
   }
   
  void
  Group::write(std::ostream& os) const {
    // This is grossly inadequate. It is also not critical for the
    // first pass.
    os << "Group for product with ID: " << provenance_->conditionsID();
  }

}
