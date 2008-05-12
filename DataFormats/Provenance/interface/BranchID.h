#ifndef DataFormats_Provenance_BranchID_h
#define DataFormats_Provenance_BranchID_h

/*----------------------------------------------------------------------
  
BranchID: A unique identifier for each branch.

----------------------------------------------------------------------*/

#include <iosfwd>
#include <string>

namespace edm {
  class BranchID {
  public:
    typedef std::string ID;
    BranchID();
    explicit BranchID(std::string const& str);
    ID id() const { return id_; }
    bool empty() const {return id_.empty();}
    void setID(std::string const& branchName) const {
      // Needed only for backward compatibility.
      ID & idR = const_cast<ID &>(id_);
      idR = branchName;
    }
    bool operator<(BranchID const& rh) const {return id_ < rh.id_;}
    bool operator>(BranchID const& rh) const {return id_ > rh.id_;}
    bool operator==(BranchID const& rh) const {return id_ == rh.id_;}
    bool operator!=(BranchID const& rh) const {return id_ != rh.id_;}
  private:
    ID id_;
  };

  std::ostream&
  operator<<(std::ostream& os, BranchID const& id);
}
#endif
