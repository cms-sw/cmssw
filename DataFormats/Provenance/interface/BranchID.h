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
    BranchID() : id_(0) { }
    explicit BranchID(std::string const& str) {
      setID(str);
    }
    void setID(std::string const& branchName);

    unsigned int id() const { return id_; }
    bool isValid() const {return id_ != 0;}

    bool operator<(BranchID const& rh) const {return id_ < rh.id_;}
    bool operator>(BranchID const& rh) const {return id_ > rh.id_;}
    bool operator==(BranchID const& rh) const {return id_ == rh.id_;}
    bool operator!=(BranchID const& rh) const {return id_ != rh.id_;}

  private:
    unsigned int id_;
  };

  std::ostream&
  operator<<(std::ostream& os, BranchID const& id);
}
#endif
