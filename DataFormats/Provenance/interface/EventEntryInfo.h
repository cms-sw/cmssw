#ifndef DataFormats_Provenance_EventEntryInfo_h
#define DataFormats_Provenance_EventEntryInfo_h

/*----------------------------------------------------------------------
  
EventEntryInfo: The event dependent portion of the description of a product
and how it came into existence, plus the product identifier.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <vector>

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"
#include "DataFormats/Provenance/interface/ProductID.h"

/*
  EventEntryInfo
*/

namespace edm {
  class EventEntryDescription;
  class EventEntryInfo {
  public:
    EventEntryInfo();
    ~EventEntryInfo();

    void write(std::ostream& os) const;

    BranchID const& branchID() const {return branchID_;}
    ProductID const& productID() const {return productID_;}
    EntryDescriptionID const& entryDescriptionID() const {return entryDescriptionID_;}

  private:

    BranchID branchID_;
    ProductID productID_;
    EntryDescriptionID entryDescriptionID_;
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
