#ifndef FWCore_Framework_WillGetIfMatch_h
#define FWCore_Framework_WillGetIfMatch_h

/** \class edm::WillGetIfMatch

This is intended to be used only by the class GetterOfProducts.
See comments in the file GetterOfProducts.h.

\author W. David Dagenhart, created 6 August, 2012

*/

#include <functional>
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm {

  template <typename T, edm::Transition Tr>
  class WillGetIfMatch {
  public:
    template <typename U>
    WillGetIfMatch(U const& match, EDConsumerBase* module) : match_(match), module_(module) {}

    EDGetTokenT<T> operator()(ProductDescription const& productDescription) {
      if (match_(productDescription)) {
        auto transition = productDescription.branchType();
        edm::InputTag tag{productDescription.moduleLabel(),
                          productDescription.productInstanceName(),
                          productDescription.processName()};
        if (transition == edm::InEvent) {
          return module_->template consumes<T, edm::InEvent, Tr>(tag);
        } else if (transition == edm::InLumi) {
          return module_->template consumes<T, edm::InLumi, Tr>(tag);
        } else if (transition == edm::InRun) {
          return module_->template consumes<T, edm::InRun, Tr>(tag);
        } else if (transition == edm::InProcess) {
          return module_->template consumes<T, edm::InProcess, Tr>(tag);
        }
      }
      return EDGetTokenT<T>{};
    }

  private:
    std::function<bool(ProductDescription const&)> match_;
    EDConsumerBase* module_;
  };
}  // namespace edm
#endif
