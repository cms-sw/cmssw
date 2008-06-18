#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/EntryDescriptionRegistry.h"
#include <ostream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  EventEntryInfo::EventEntryInfo() :
    branchID_(),
    productID_(),
    productStatus_(productstatus::uninitialized()),
    entryDescriptionID_(),
    entryDescriptionPtr_()
  {}

  EventEntryInfo::EventEntryInfo(BranchID const& bid) :
    branchID_(bid),
    productID_(),
    productStatus_(productstatus::uninitialized()),
    entryDescriptionID_(),
    entryDescriptionPtr_()
  {}

   EventEntryInfo::EventEntryInfo(BranchID const& bid,
				    ProductStatus status,
				    ProductID const& pid) :
    branchID_(bid),
    productID_(pid),
    productStatus_(status),
    entryDescriptionID_(),
    entryDescriptionPtr_()
  {}

   EventEntryInfo::EventEntryInfo(BranchID const& bid,
				    ProductStatus status,
				    ProductID const& pid,
				    EntryDescriptionID const& edid) :
    branchID_(bid),
    productID_(pid),
    productStatus_(status),
    entryDescriptionID_(edid),
    entryDescriptionPtr_()
  {}

   EventEntryInfo::EventEntryInfo(BranchID const& bid,
				    ProductStatus status,
				    ProductID const& pid,
				    boost::shared_ptr<EntryDescription> edPtr) :
    branchID_(bid),
    productID_(pid),
    productStatus_(status),
    entryDescriptionID_(edPtr->id()),
    entryDescriptionPtr_(edPtr)
  { EntryDescriptionRegistry::instance()->insertMapped(*edPtr);}

  EventEntryInfo::EventEntryInfo(BranchID const& bid,
		   ProductStatus status,
		   ModuleDescriptionID const& mdid,
		   ProductID const& pid,
		   std::vector<BranchID> const& parents) :
    branchID_(bid),
    productID_(pid),
    productStatus_(status),
    entryDescriptionID_(),
    entryDescriptionPtr_(new EntryDescription) {
      entryDescriptionPtr_->parents_ = parents;
      entryDescriptionPtr_->moduleDescriptionID_ = mdid;
      entryDescriptionID_ = entryDescriptionPtr_->id();
      EntryDescriptionRegistry::instance()->insertMapped(*entryDescriptionPtr_);
  }

  EntryDescription const &
  EventEntryInfo::entryDescription() const {
    if (!entryDescriptionPtr_) {
      entryDescriptionPtr_.reset(new EntryDescription);
      EntryDescriptionRegistry::instance()->getMapped(entryDescriptionID_, *entryDescriptionPtr_);
    }
    return *entryDescriptionPtr_;
  }

  void
  EventEntryInfo::setPresent() {
    if (productstatus::present(productStatus())) return;
    assert(productstatus::unknown(productStatus()));
    setStatus(productstatus::present());
  }

  void
  EventEntryInfo::setNotPresent() {
    if (productstatus::neverCreated(productStatus())) return;
    if (productstatus::dropped(productStatus())) return;
    assert(productstatus::unknown(productStatus()));
    setStatus(productstatus::neverCreated());
  }

  void
  EventEntryInfo::write(std::ostream& os) const {
    os << "branch ID = " << branchID() << '\n';
    os << "product ID = " << productID() << '\n';
    os << "product status = " << productStatus() << '\n';
    os << "entry description ID = " << entryDescriptionID() << '\n';
  }
    
  bool
  operator==(EventEntryInfo const& a, EventEntryInfo const& b) {
    return
      a.branchID() == b.branchID()
      && a.productID() == b.productID()
      && a.productStatus() == b.productStatus()
      && a.entryDescriptionID() == b.entryDescriptionID();
  }
}
