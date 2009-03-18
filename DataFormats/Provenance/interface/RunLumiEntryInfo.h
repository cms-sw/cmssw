#ifndef DataFormats_Provenance_RunLumiEntryInfo_h
#define DataFormats_Provenance_RunLumiEntryInfo_h

/*----------------------------------------------------------------------
  
RunLumiEntryInfo: The event dependent portion of the description of a product
and how it came into existence, plus the product identifier and the status.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <vector>

#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

/*
  RunLumiEntryInfo
*/
namespace edm {
  class RunLumiEntryInfo {
  public:
    typedef std::vector<RunLumiEntryInfo> EntryInfoVector;
    RunLumiEntryInfo();
    explicit RunLumiEntryInfo(BranchID const& bid);
    explicit RunLumiEntryInfo(ProductProvenance const& ei);
    RunLumiEntryInfo(BranchID const& bid,
		    ProductStatus status);
    RunLumiEntryInfo(BranchID const& bid,
		    ProductStatus status,
		    std::vector<BranchID> const& parents);

    RunLumiEntryInfo(BranchID const& bid,
		    ProductStatus status,
		    EntryDescriptionID const& edid);

    ~RunLumiEntryInfo() {}

    ProductProvenance makeProductProvenance() const;

    void write(std::ostream& os) const;

    BranchID const& branchID() const {return branchID_;}
    ProductStatus const& productStatus() const {return productStatus_;}
    void setStatus(ProductStatus status) {productStatus_ = status;}
    void setPresent();
    void setNotPresent();

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
