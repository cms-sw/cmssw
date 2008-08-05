#include "DataFormats/Provenance/interface/Provenance.h"

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {

  Provenance::Provenance(BranchDescription const& p) :
    branchDescription_(p),
    branchEntryInfoPtr_() {
  }

  Provenance::Provenance(ConstBranchDescription const& p) :
    branchDescription_(p),
    branchEntryInfoPtr_() {
  }

  Provenance::Provenance(BranchDescription const& p, boost::shared_ptr<EventEntryInfo> ei) :
    branchDescription_(p),
    branchEntryInfoPtr_(ei)
  { }

  Provenance::Provenance(ConstBranchDescription const& p, boost::shared_ptr<EventEntryInfo> ei) :
    branchDescription_(p),
    branchEntryInfoPtr_(ei)
  { }

  Provenance::Provenance(BranchDescription const& p, boost::shared_ptr<RunLumiEntryInfo> ei) :
    branchDescription_(p),
    branchEntryInfoPtr_(boost::shared_ptr<EventEntryInfo>(
      new EventEntryInfo(ei->branchID(), ei->productStatus(), ei->moduleDescriptionID()))) {
  }

  Provenance::Provenance(ConstBranchDescription const& p, boost::shared_ptr<RunLumiEntryInfo> ei) :
    branchDescription_(p),
    branchEntryInfoPtr_(boost::shared_ptr<EventEntryInfo>(
      new EventEntryInfo(ei->branchID(), ei->productStatus(), ei->moduleDescriptionID()))) {
  }

  void
  Provenance::setEventEntryInfo(boost::shared_ptr<EventEntryInfo> bei) const {
    assert(branchEntryInfoPtr_.get() == 0);
    branchEntryInfoPtr_ = bei;
  }

  boost::shared_ptr<EventEntryInfo>
  Provenance::resolve () const {
    boost::shared_ptr<EventEntryInfo> prov = store_->branchToEntryInfo(branchDescription_.branchID());
    setEventEntryInfo(prov);
    return prov;
}


  void
  Provenance::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    product().write(os);
    branchEntryInfo().write(os);
  }

    
  bool operator==(Provenance const& a, Provenance const& b) {
    return
      a.product() == b.product()
      && a.entryDescription() == b.entryDescription();
  }

}

