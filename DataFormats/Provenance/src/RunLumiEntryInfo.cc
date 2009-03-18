#include "DataFormats/Provenance/interface/ProductProvenance.h"
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
    productStatus_(productstatus::uninitialized())
  {}

  RunLumiEntryInfo::RunLumiEntryInfo(ProductProvenance const& ei) :
    branchID_(ei.branchID()),
    productStatus_(ei.productStatus())
  {}

  RunLumiEntryInfo::RunLumiEntryInfo(BranchID const& bid) :
    branchID_(bid),
    productStatus_(productstatus::uninitialized())
  {}

   RunLumiEntryInfo::RunLumiEntryInfo(BranchID const& bid,
				    ProductStatus status) :
    branchID_(bid),
    productStatus_(status)
  {}

   RunLumiEntryInfo::RunLumiEntryInfo(BranchID const& bid,
				    ProductStatus status,
				    EntryDescriptionID const& edid) :
    branchID_(bid),
    productStatus_(status) {
     EventEntryDescription ed;
     EntryDescriptionRegistry::instance()->getMapped(edid, ed);
  } 

   // The last argument is ignored.
   // It is used for backward compatibility.
   RunLumiEntryInfo::RunLumiEntryInfo(BranchID const& bid,
                                  ProductStatus status,
                                  std::vector<BranchID> const&) :
    branchID_(bid),
    productStatus_(status)
  {}


  ProductProvenance
  RunLumiEntryInfo::makeProductProvenance() const {
    return ProductProvenance(branchID_, productStatus_);
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
  }
    
  bool
  operator==(RunLumiEntryInfo const& a, RunLumiEntryInfo const& b) {
    return
      a.branchID() == b.branchID()
      && a.productStatus() == b.productStatus();
  }
}
