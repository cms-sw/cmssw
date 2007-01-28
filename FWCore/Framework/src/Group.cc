/*----------------------------------------------------------------------
$Id: Group.cc,v 1.13 2006/08/24 22:03:12 wmtan Exp $
----------------------------------------------------------------------*/

#include <string>
#include "FWCore/Framework/src/Group.h"

namespace edm
{
  Group::Group(std::auto_ptr<Provenance> prov,
	       bool acc, bool onDemand) :
    product_(),
    provenance_(prov.release()),
    accessible_(acc),
    onDemand_(onDemand) {
  }

  Group::Group(std::auto_ptr<EDProduct> edp,
	       std::auto_ptr<Provenance> prov,
	       bool acc) :
    product_(edp.release()),
    provenance_(prov.release()),
    accessible_(acc),
    onDemand_(false) {
  }

  Group::~Group() {
    delete product_;
    delete provenance_;
  }

  bool 
  Group::productAvailable() const { 
      return 
	accessible_ and
	((provenance_->creatorStatus() == BranchEntryDescription::Success) or onDemand_);
  }

  bool 
  Group::provenanceAvailable() const { 
      return accessible_ and not onDemand_;
  }

  void 
  Group::setProduct(std::auto_ptr<EDProduct> prod) const {
    assert (product() == 0);
    product_ = prod.release();  // Group takes ownership
  }
  
  void  
  Group::swap(Group& other) {
    std::swap(product_,other.product_);
    std::swap(provenance_,other.provenance_);
    std::swap(accessible_, other.accessible_);
    std::swap(onDemand_, other.onDemand_);
  }

  void
  Group::write(std::ostream& os) const {
    // This is grossly inadequate. It is also not critical for the
    // first pass.
    os << std::string("Group for product with ID: ") << provenance_->productID();
  }

}
