#ifndef DataFormats_Provenance_EventEntryInfo_h
#define DataFormats_Provenance_EventEntryInfo_h

/*----------------------------------------------------------------------
  
EventEntryInfo: The event dependent portion of the description of a product
and how it came into existence, plus the product identifier and the status.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <vector>

#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

/*
  EventEntryInfo
*/

namespace edm {
  class EventEntryInfo {
  public:
    EventEntryInfo();
    explicit EventEntryInfo(BranchID const& bid);
    EventEntryInfo(BranchID const& bid,
		    ProductStatus status,
		    ProductID const& pid);
    EventEntryInfo(BranchID const& bid,
		    ProductStatus status,
		    ProductID const& pid,
		    boost::shared_ptr<EventEntryDescription> edPtr);
    EventEntryInfo(BranchID const& bid,
		    ProductStatus status,
		    ProductID const& pid,
		    EntryDescriptionID const& edid);

    EventEntryInfo(BranchID const& bid,
		   ProductStatus status,
		   ModuleDescriptionID const& mdid,
		   ProductID const& pid = ProductID(),
		   std::vector<BranchID> const& parents = std::vector<BranchID>());

    ~EventEntryInfo() {}

    void write(std::ostream& os) const;

    BranchID const& branchID() const {return branchID_;}
    ProductID const& productID() const {return productID_;}
    ProductStatus const& productStatus() const {return productStatus_;}
    EntryDescriptionID const& entryDescriptionID() const {return entryDescriptionID_;}
    EventEntryDescription const& entryDescription() const;
    void setStatus(ProductStatus status) {productStatus_ = status;}
    void setPresent();
    void setNotPresent();

  private:
    BranchID branchID_;
    ProductID productID_;
    ProductStatus productStatus_;
    EntryDescriptionID entryDescriptionID_;
    mutable boost::shared_ptr<EventEntryDescription> entryDescriptionPtr_;
  };

  inline
  bool
  operator < (EventEntryInfo const& a, EventEntryInfo const& b) {
    return a.branchID() < b.branchID();
  }
  
  inline
  std::ostream&
  operator<<(std::ostream& os, EventEntryInfo const& p) {
    p.write(os);
    return os;
  }

  // Only the 'salient attributes' are testing in equality comparison.
  bool operator==(EventEntryInfo const& a, EventEntryInfo const& b);
  inline bool operator!=(EventEntryInfo const& a, EventEntryInfo const& b) { return !(a==b); }

  typedef std::vector<EventEntryInfo> EventEntryInfoVector;
}
#endif
