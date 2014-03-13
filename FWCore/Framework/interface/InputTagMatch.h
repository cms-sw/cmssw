#ifndef FWCore_Framework_InputTagMatch_h
#define FWCore_Framework_InputTagMatch_h

/** \class edm::InputTagMatch

This is intended to be used with the class GetterOfProducts.
See comments in the file GetterOfProducts.h for a description.

\author V. Adler

*/

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"

#include <string>

namespace edm {

   class InputTagMatch {
   public:

      InputTagMatch(std::string const& encodedInputTag) : inputTag_(encodedInputTag) { }

      bool operator()(edm::BranchDescription const& branchDescription) {
         bool result(true);
         bool match(false);
         if (!inputTag_.label().empty()) {
           match = true;
           result = (result && branchDescription.moduleLabel() == inputTag_.label());
         }
         if (!inputTag_.instance().empty()) {
           match = true;
           result = (result && branchDescription.productInstanceName() == inputTag_.instance());
         }
         if (!inputTag_.process().empty()) {
           match = true;
           result = (result && branchDescription.processName() == inputTag_.process());
         }
         if (match) return result;
         return false;
      }

   private:
      edm::InputTag inputTag_;
   };
}
#endif
