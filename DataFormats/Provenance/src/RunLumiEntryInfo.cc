#include "DataFormats/Provenance/interface/RunLumiEntryInfo.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"
#include "DataFormats/Provenance/interface/EventEntryDescription.h"
#include "DataFormats/Provenance/interface/EntryDescriptionRegistry.h"
#include <ostream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  RunLumiEntryInfo::RunLumiEntryInfo() :
    branchID_(),
    productStatus_(productstatus::uninitialized()),
    moduleDescriptionID_()
  {}

  RunLumiEntryInfo::RunLumiEntryInfo(BranchID const& bid) :
    branchID_(bid),
    productStatus_(productstatus::uninitialized()),
    moduleDescriptionID_()
  {}

   RunLumiEntryInfo::RunLumiEntryInfo(BranchID const& bid,
				    ProductStatus status) :
    branchID_(bid),
    productStatus_(status),
    moduleDescriptionID_()
  {}

   // The last two arguments are ignored.
   // They are used for backward compatibility.
   RunLumiEntryInfo::RunLumiEntryInfo(BranchID const& bid,
				    ProductStatus status,
				    ModuleDescriptionID const& mid,
		    		    ProductID const&,
				    std::vector<BranchID> const&) :
    branchID_(bid),
    productStatus_(status),
    moduleDescriptionID_(mid)
  {} 

   RunLumiEntryInfo::RunLumiEntryInfo(BranchID const& bid,
				    ProductStatus status,
		    		    ProductID const&,
				    EntryDescriptionID const& edid) :
    branchID_(bid),
    productStatus_(status),
    moduleDescriptionID_() {
     EventEntryDescription ed;
     EntryDescriptionRegistry::instance()->getMapped(edid, ed);
     moduleDescriptionID_ = ed.moduleDescriptionID();
  } 

  void
  RunLumiEntryInfo::setPresent() {
    if (productstatus::present(productStatus())) return;
    assert(productstatus::unknown(productStatus()));
    setStatus(productstatus::present());
  }

  void
  RunLumiEntryInfo::setNotPresent() {
    if (productstatus::neverCreated(productStatus())) return;
    assert(productstatus::unknown(productStatus()));
    setStatus(productstatus::neverCreated());
  }

  void
  RunLumiEntryInfo::write(std::ostream& os) const {
    os << "branch ID = " << branchID() << '\n';
    os << "product status = " << productStatus() << '\n';
    os << "module description ID = " << moduleDescriptionID() << '\n';
  }
    
  bool
  operator==(RunLumiEntryInfo const& a, RunLumiEntryInfo const& b) {
    return
      a.branchID() == b.branchID()
      && a.productStatus() == b.productStatus()
      && a.moduleDescriptionID() == b.moduleDescriptionID();
  }
}
