#ifndef FWCore_Framework_ModuleLabelMatch_h
#define FWCore_Framework_ModuleLabelMatch_h

/** \class edm::ModuleLabelMatch

This is intended to be used with the class GetterOfProducts.
See comments in the file GetterOfProducts.h for a description.

\author W. David Dagenhart, created 6 August, 2012

*/

#include "DataFormats/Provenance/interface/ProductDescription.h"

#include <string>

namespace edm {

  class ModuleLabelMatch {
  public:
    ModuleLabelMatch(std::string const& moduleLabel) : moduleLabel_(moduleLabel) {}

    bool operator()(edm::ProductDescription const& productDescription) {
      return productDescription.moduleLabel() == moduleLabel_;
    }

  private:
    std::string moduleLabel_;
  };
}  // namespace edm
#endif
