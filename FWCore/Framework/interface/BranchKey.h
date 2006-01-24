#ifndef Framework_BranchKey_h
#define Framework_BranchKey_h

/*----------------------------------------------------------------------
  
BranchKey: The key used to identify a Group in the EventPrincipal. The
name of the branch to which the related data product will be written
is determined entirely from the BranchKey.

$Id: BranchKey.h,v 1.11 2005/10/26 22:33:52 wmtan Exp $

----------------------------------------------------------------------*/
#include <iosfwd>
#include <string>
#include <utility>

namespace edm {
  class BranchDescription;

  struct BranchKey {
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

    std::string friendlyClassName_;
    std::string moduleLabel_;
    std::string productInstanceName_;
    std::string processName_; // ???
  };

  inline
  bool 
  operator<(const BranchKey& a, const BranchKey& b) {
      return 
	a.friendlyClassName_ < b.friendlyClassName_ ? true :
	a.friendlyClassName_ > b.friendlyClassName_ ? false :
	a.moduleLabel_ < b.moduleLabel_ ? true :
	a.moduleLabel_ > b.moduleLabel_ ? false :
	a.productInstanceName_ < b.productInstanceName_ ? true :
	a.productInstanceName_ > b.productInstanceName_ ? false :
	a.processName_ < b.processName_ ? true :
	false;
  }

  inline
  bool 
  operator==(const BranchKey& a, const BranchKey& b) {
    return !(a < b || b < a);
  }

  std::ostream&
  operator<<(std::ostream& os, const BranchKey& bk);
}
#endif
