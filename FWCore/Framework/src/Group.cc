/*----------------------------------------------------------------------
$Id: Group.cc,v 1.14 2007/01/28 05:40:57 wmtan Exp $
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

  bool
  Group::replace(Group& g) {
    if(onDemand_) {
      // The old one is a "placeholder" group for unscheduled processing.
      // This new one is the one generated 'unscheduled'.
      // NOTE: other API's of DataBlockImpl do NOT give out the Provenance*
      // to "onDemand" groups, so no need to preserve the old Provenance.
      this->swap(g);
      return true;
    }
    return false;
  }

  void
  Group::write(std::ostream& os) const {
    // This is grossly inadequate. It is also not critical for the
    // first pass.
    os << std::string("Group for product with ID: ") << provenance_->productID();
  }

}
