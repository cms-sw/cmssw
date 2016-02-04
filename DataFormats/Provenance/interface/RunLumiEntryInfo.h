#ifndef DataFormats_Provenance_RunLumiEntryInfo_h
#define DataFormats_Provenance_RunLumiEntryInfo_h

/*----------------------------------------------------------------------
  
RunLumiEntryInfo: The event dependent portion of the description of a product
and how it came into existence, plus the product identifier and the status.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <vector>

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"

/*
  RunLumiEntryInfo
*/
namespace edm {
  class RunLumiEntryInfo {
  public:
    RunLumiEntryInfo();
    ~RunLumiEntryInfo();

    void write(std::ostream& os) const;

    BranchID const& branchID() const {return branchID_;}
    ProductStatus const& productStatus() const {return productStatus_;}

  private:
    BranchID branchID_;
    ProductStatus productStatus_;
  };

  inline
  bool
  operator < (RunLumiEntryInfo const& a, RunLumiEntryInfo const& b) {
    return a.branchID() < b.branchID();
  }
  
  inline
  std::ostream&
  operator<<(std::ostream& os, RunLumiEntryInfo const& p) {
    p.write(os);
    return os;
  }

  // Only the 'salient attributes' are testing in equality comparison.
  bool operator==(RunLumiEntryInfo const& a, RunLumiEntryInfo const& b);
  inline bool operator!=(RunLumiEntryInfo const& a, RunLumiEntryInfo const& b) { return !(a==b); }

  typedef RunLumiEntryInfo LumiEntryInfo;
  typedef RunLumiEntryInfo RunEntryInfo;
}
#endif
