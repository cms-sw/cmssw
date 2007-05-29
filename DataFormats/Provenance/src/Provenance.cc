#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: Provenance.cc,v 1.4 2007/05/10 22:46:54 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  Provenance::Provenance(BranchDescription const& p) :
    product_(ConstBranchDescription(p)),
    event_(),
    store_()
  { }

  Provenance::Provenance(ConstBranchDescription const& p) :
    product_(p),
    event_(),
    store_()
  { }

  Provenance::Provenance(BranchDescription const& p, BranchEntryDescription::CreatorStatus const& status) :
    product_(ConstBranchDescription(p)),
    event_(new BranchEntryDescription(p.productID(), status)),
    store_()
  { }

  Provenance::Provenance(ConstBranchDescription const& p, BranchEntryDescription::CreatorStatus const& status) :
    product_(p),
    event_(new BranchEntryDescription(p.productID(), status)),
    store_()
  { }

  Provenance::Provenance(BranchDescription const& p, boost::shared_ptr<BranchEntryDescription> e) :
    product_(ConstBranchDescription(p)),
    event_(e),
    store_()
  { }

  Provenance::Provenance(ConstBranchDescription const& p, boost::shared_ptr<BranchEntryDescription> e) :
    product_(p),
    event_(e),
    store_()
  { }

 Provenance::Provenance(BranchDescription const& p, BranchEntryDescription const& e) :
    product_(ConstBranchDescription(p)),
    event_(new BranchEntryDescription(e)),
    store_()
  { }

 Provenance::Provenance(ConstBranchDescription const& p, BranchEntryDescription const& e) :
    product_(p),
    event_(new BranchEntryDescription(e)),
    store_()
  { }

  void
  Provenance::setEvent(boost::shared_ptr<BranchEntryDescription> e) const {
    assert(event_.get() == 0);
    event_ = e;
  }

  BranchEntryDescription const& 
  Provenance::resolve () const {
    std::auto_ptr<BranchEntryDescription> prov = store_->getProvenance(BranchKey(product_));
    setEvent(boost::shared_ptr<BranchEntryDescription>(prov.release()));
    return *event_; 
  }

  void
  Provenance::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    product().write(os);
    event().write(os);
  }

    
  bool operator==(Provenance const& a, Provenance const& b) {
    return
      a.product() == b.product()
      && a.event() == b.event();
  }

}

