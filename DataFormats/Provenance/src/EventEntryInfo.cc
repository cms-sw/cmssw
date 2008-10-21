#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/EntryDescriptionRegistry.h"
#include <ostream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  EventEntryInfo::Transients::Transients() :
    moduleDescriptionID_(),
    entryDescriptionPtr_(),
    noEntryDescription_(false)
  {}

  EventEntryInfo::EventEntryInfo() :
    branchID_(),
    productID_(),
    productStatus_(productstatus::uninitialized()),
    entryDescriptionID_(),
    transients_()
  {}

  EventEntryInfo::EventEntryInfo(BranchID const& bid) :
    branchID_(bid),
    productID_(),
    productStatus_(productstatus::uninitialized()),
    entryDescriptionID_(),
    transients_()
  {}

   EventEntryInfo::EventEntryInfo(BranchID const& bid,
				    ProductStatus status,
				    ProductID const& pid) :
    branchID_(bid),
    productID_(pid),
    productStatus_(status),
    entryDescriptionID_(),
    transients_()
  {}

   EventEntryInfo::EventEntryInfo(BranchID const& bid,
				    ProductStatus status,
				    ProductID const& pid,
				    EntryDescriptionID const& edid) :
    branchID_(bid),
    productID_(pid),
    productStatus_(status),
    entryDescriptionID_(edid),
    transients_()
  {}

   EventEntryInfo::EventEntryInfo(BranchID const& bid,
				    ProductStatus status,
				    ProductID const& pid,
				    boost::shared_ptr<EventEntryDescription> edPtr) :
    branchID_(bid),
    productID_(pid),
    productStatus_(status),
    entryDescriptionID_(edPtr->id()),
    transients_() {
       moduleDescriptionID() = edPtr->moduleDescriptionID();
       entryDescriptionPtr() = edPtr;
       EntryDescriptionRegistry::instance()->insertMapped(*edPtr);
  }

  EventEntryInfo::EventEntryInfo(BranchID const& bid,
		   ProductStatus status,
		   ModuleDescriptionID const& mdid,
		   ProductID const& pid,
		   std::vector<BranchID> const& parents) :
    branchID_(bid),
    productID_(pid),
    productStatus_(status),
    entryDescriptionID_(),
    transients_() {
      moduleDescriptionID() = mdid;
      entryDescriptionPtr() = boost::shared_ptr<EventEntryDescription>(new EventEntryDescription);
      entryDescriptionPtr()->parents() = parents;
      entryDescriptionPtr()->moduleDescriptionID() = mdid;
      entryDescriptionID_ = entryDescriptionPtr()->id();
      EntryDescriptionRegistry::instance()->insertMapped(*entryDescriptionPtr());
  }

  EventEntryInfo::EventEntryInfo(BranchID const& bid,
		   ProductStatus status,
		   ModuleDescriptionID const& mdid) :
    branchID_(bid),
    productID_(),
    productStatus_(status),
    entryDescriptionID_(),
    transients_() {
      moduleDescriptionID() = mdid;
      noEntryDescription() = true;
    }

  EventEntryInfo
  EventEntryInfo::makeEntryInfo() const {
    return *this;
  }

  EventEntryDescription const &
  EventEntryInfo::entryDescription() const {
    if (!entryDescriptionPtr()) {
      entryDescriptionPtr().reset(new EventEntryDescription);
      EntryDescriptionRegistry::instance()->getMapped(entryDescriptionID_, *entryDescriptionPtr());
      moduleDescriptionID() = entryDescriptionPtr()->moduleDescriptionID();
    }
    return *entryDescriptionPtr();
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
    os << "product status = " << static_cast<int>(productStatus()) << '\n';
    if (noEntryDescription()) {
      os << "module description ID = " << moduleDescriptionID() << '\n';
    } else {
      os << "entry description ID = " << entryDescriptionID() << '\n';
    }
  }
    
  bool
  operator==(EventEntryInfo const& a, EventEntryInfo const& b) {
    if (a.noEntryDescription() != b.noEntryDescription()) return false;
    if (a.noEntryDescription()) {
      return
        a.branchID() == b.branchID()
        && a.productID() == b.productID()
        && a.productStatus() == b.productStatus()
        && a.moduleDescriptionID() == b.moduleDescriptionID();
    }
    return
      a.branchID() == b.branchID()
      && a.productID() == b.productID()
      && a.productStatus() == b.productStatus()
      && a.entryDescriptionID() == b.entryDescriptionID();
  }
}
