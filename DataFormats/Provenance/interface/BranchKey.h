#ifndef DataFormats_Provenance_BranchKey_h
#define DataFormats_Provenance_BranchKey_h

/*----------------------------------------------------------------------
  
BranchKey: The key used to identify a Product in the EventPrincipal. The
name of the branch to which the related data product will be written
is determined entirely from the BranchKey.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <string>

namespace edm {
  class BranchDescription;
  class ConstBranchDescription;

  class BranchKey {
  public:
    BranchKey() : friendlyClassName_(), moduleLabel_(), productInstanceName_(), processName_()
    {}

    BranchKey(std::string const& cn, std::string const& ml,
        std::string const& pin, std::string const& pn) :
      friendlyClassName_(cn), 
      moduleLabel_(ml), 
      productInstanceName_(pin), 
      processName_(pn) 
    {}

    explicit BranchKey(BranchDescription const& desc);
    explicit BranchKey(ConstBranchDescription const& desc);

    std::string const& friendlyClassName() const {return friendlyClassName_;}
    std::string const& moduleLabel() const {return moduleLabel_;}
    std::string const& productInstanceName() const {return productInstanceName_;}
    std::string const& processName() const {return processName_;}

  private:
    std::string friendlyClassName_;
    std::string moduleLabel_;
    std::string productInstanceName_;
    std::string processName_;
  };

  inline
  bool 
  operator<(BranchKey const& a, BranchKey const& b) {
      return 
	a.friendlyClassName() < b.friendlyClassName() ? true :
	a.friendlyClassName() > b.friendlyClassName() ? false :
	a.moduleLabel() < b.moduleLabel() ? true :
	a.moduleLabel() > b.moduleLabel() ? false :
	a.productInstanceName() < b.productInstanceName() ? true :
	a.productInstanceName() > b.productInstanceName() ? false :
	a.processName() < b.processName() ? true :
	false;
  }

  inline
  bool 
  operator==(BranchKey const& a, BranchKey const& b) {
    return !(a < b || b < a);
  }

  inline
  bool 
  operator!=(BranchKey const& a, BranchKey const& b) {
    return !(a == b);
  }

  std::ostream&
  operator<<(std::ostream& os, BranchKey const& bk);
}
#endif
