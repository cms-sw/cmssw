#ifndef FWCore_Framework_WillGetIfMatch_h
#define FWCore_Framework_WillGetIfMatch_h

/** \class edm::WillGetIfMatch

This is intended to be used only by the class GetterOfProducts.
See comments in the file GetterOfProducts.h.

\author W. David Dagenhart, created 6 August, 2012

*/

#include <functional>
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm {

  class BranchDescription;
  class EDConsumerBase;

  template<typename T>
  class WillGetIfMatch {
  public:

    template <typename U>
    WillGetIfMatch(U const& match, EDConsumerBase* module):
      match_(match),
      module_(module) {
    }
    
    EDGetTokenT<T>  operator()(BranchDescription const& branchDescription) {
      if (match_(branchDescription)){
        auto transition = branchDescription.branchType();
        edm::InputTag tag{branchDescription.moduleLabel(),
                          branchDescription.productInstanceName(),
                          branchDescription.processName()};
        if(transition == edm::InEvent) {
          return module_->template consumes<T>(tag);
        } else if(transition == edm::InLumi) {
          return module_->template consumes<T,edm::InLumi>(tag);
        } else if(transition == edm::InRun) {
          return module_->template consumes<T,edm::InRun>(tag);
        }
      }
      return EDGetTokenT<T>{};
    }
    
  private:
    std::function<bool(BranchDescription const&)> match_;
    EDConsumerBase* module_;
  };
}
#endif
