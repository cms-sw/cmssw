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
    typedef unsigned int value_type;
    BranchID() : id_(0) { }
    explicit BranchID(std::string const& branchName) : id_(toID(branchName)) {
    }
    explicit BranchID(value_type theID) : id_(theID) {
    }
    void setID(std::string const& branchName) {id_ = toID(branchName);}
    unsigned int id() const { return id_; }
    bool isValid() const {return id_ != 0;}

    bool operator<(BranchID const& rh) const {return id_ < rh.id_;}
    bool operator>(BranchID const& rh) const {return id_ > rh.id_;}
    bool operator==(BranchID const& rh) const {return id_ == rh.id_;}
    bool operator!=(BranchID const& rh) const {return id_ != rh.id_;}

  private:
    static value_type toID(std::string const& branchName);
    value_type id_;
  };
  

  std::ostream&
  operator<<(std::ostream& os, BranchID const& id);
}
#endif
