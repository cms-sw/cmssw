/*----------------------------------------------------------------------
$Id: Group.cc,v 1.12 2006/07/06 19:11:43 wmtan Exp $
----------------------------------------------------------------------*/

#include <string>
#include "FWCore/Framework/src/Group.h"

namespace edm
{
  Group::Group(std::auto_ptr<Provenance> prov,
	       bool acc) :
    product_(),
    provenance_(prov.release()),
    accessible_(acc) {
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
    os << std::string("Group for product with ID: ") << provenance_->productID();
  }

}
