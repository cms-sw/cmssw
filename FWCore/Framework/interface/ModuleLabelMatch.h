#ifndef FWCore_Framework_ModuleLabelMatch_h
#define FWCore_Framework_ModuleLabelMatch_h

/** \class edm::ModuleLabelMatch

This is intended to be used with the class GetterOfProducts.
See comments in the file GetterOfProducts.h for a description.

\author W. David Dagenhart, created 6 August, 2012

*/

#include "DataFormats/Provenance/interface/BranchDescription.h"

#include <string>

namespace edm {

   class ModuleLabelMatch {
   public:

      ModuleLabelMatch(std::string const& moduleLabel) : moduleLabel_(moduleLabel) { }

      bool operator()(edm::BranchDescription const& branchDescription) {
         return branchDescription.moduleLabel() == moduleLabel_;
      }

   private:
      std::string moduleLabel_;
   };
}
#endif
