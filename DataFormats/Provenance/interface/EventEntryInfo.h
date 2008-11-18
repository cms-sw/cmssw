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
    typedef std::vector<EventEntryInfo> EntryInfoVector;
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
		   ProductID const& pid,
		   std::vector<BranchID> const& parents = std::vector<BranchID>());

    EventEntryInfo(BranchID const& bid,
		   ProductStatus status,
		   ModuleDescriptionID const& mdid);

    ~EventEntryInfo() {}

    EventEntryInfo makeEntryInfo() const;

    void write(std::ostream& os) const;

    BranchID const& branchID() const {return branchID_;}
    ProductID const& productID() const {return productID_;}
    ProductStatus const& productStatus() const {return productStatus_;}
    EntryDescriptionID const& entryDescriptionID() const {return entryDescriptionID_;}
    ModuleDescriptionID const& moduleDescriptionID() const {return moduleDescriptionID_;}
    EventEntryDescription const& entryDescription() const;
    bool noEntryDescription() const {return noEntryDescription_;}
    void setStatus(ProductStatus status) {productStatus_ = status;}
    void setPresent();
    void setNotPresent();
    void setModuleDescriptionID(ModuleDescriptionID const& mdid) {moduleDescriptionID_ = mdid;}

  private:
    BranchID branchID_;
    ProductID productID_;
    ProductStatus productStatus_;
    EntryDescriptionID entryDescriptionID_;
    mutable ModuleDescriptionID moduleDescriptionID_; //transient
    mutable boost::shared_ptr<EventEntryDescription> entryDescriptionPtr_; //! transient
    bool noEntryDescription_; //!transient
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
