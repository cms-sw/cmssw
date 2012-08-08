#ifndef FWCore_Framework_WillGetIfMatch_h
#define FWCore_Framework_WillGetIfMatch_h

/** \class edm::WillGetIfMatch

This is intended to be used only by the class GetterOfProducts.
See comments in the file GetterOfProducts.h.

\author W. David Dagenhart, created 6 August, 2012

*/

#include <functional>

namespace edm {

  class BranchDescription;

  template<typename T, typename M>
  class WillGetIfMatch {
  public:

    template <typename U>
    WillGetIfMatch(U const& match, M* module):
      match_(match),
      module_(module) {
    }
    
    bool operator()(BranchDescription const& branchDescription) {
      if (match_(branchDescription)){

        // We plan to implement a call to a function that
        // registers the products a module will get, but
        // this has not been implemented yet. This is
        // where that function would get called when it
        // is implemented, automatically registering
        // the gets for the module. (Creating a place to call
        // this function is the main reason for the existence
        // of this class).
        // module_->template willGet<T>(edm::makeInputTag(branchDescription));

        return true;
      }
      return false;
    }
    
  private:
    std::function<bool(BranchDescription const&)> match_;
    M* module_;    
  };
}
#endif
