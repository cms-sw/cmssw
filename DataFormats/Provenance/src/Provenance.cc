#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchKey.h"

/*----------------------------------------------------------------------

$Id: Provenance.cc,v 1.8 2008/01/30 00:17:54 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  Provenance::Provenance(BranchDescription const& p, bool present) :
    product_(ConstBranchDescription(p)),
    event_(),
    isPresent_(present),
    store_()
  { }

  Provenance::Provenance(ConstBranchDescription const& p, bool present) :
    product_(p),
    event_(),
    isPresent_(present),
    store_()
  { }

  Provenance::Provenance(BranchDescription const& p, boost::shared_ptr<EntryDescription> e, bool present) :
    product_(ConstBranchDescription(p)),
    event_(e),
    isPresent_(present),
    store_()
  { }

  Provenance::Provenance(ConstBranchDescription const& p, boost::shared_ptr<EntryDescription> e, bool present) :
    product_(p),
    event_(e),
    isPresent_(present),
    store_()
  { }

 Provenance::Provenance(BranchDescription const& p, EntryDescription const& e, bool present) :
    product_(ConstBranchDescription(p)),
    event_(new EntryDescription(e)),
    isPresent_(present),
    store_()
  { }

 Provenance::Provenance(ConstBranchDescription const& p, EntryDescription const& e, bool present) :
    product_(p),
    event_(new EntryDescription(e)),
    isPresent_(present),
    store_()
  { }

  void
  Provenance::setEvent(boost::shared_ptr<EntryDescription> e) const {
    assert(event_.get() == 0);
    event_ = e;
  }

  EntryDescription const& 
  Provenance::resolve () const {
    std::auto_ptr<EntryDescription> prov = store_->getProvenance(BranchKey(product_));
    setEvent(boost::shared_ptr<EntryDescription>(prov.release()));
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

