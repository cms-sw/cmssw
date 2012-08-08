#ifndef FWCore_Framework_ProcessMatch_h
#define FWCore_Framework_ProcessMatch_h

/** \class edm::ProcessMatch

This is intended to be used with the class GetterOfProducts.
See comments in the file GetterOfProducts.h for a description.

\author W. David Dagenhart, created 6 August, 2012

*/

#include "DataFormats/Provenance/interface/BranchDescription.h"

#include <string>

namespace edm {

   class ProcessMatch {
   public:

      ProcessMatch(std::string const& processName) : processName_(processName) { }

      bool operator()(edm::BranchDescription const& branchDescription) {
         return branchDescription.processName() == processName_ || processName_ == "*";
      }

   private:
      std::string processName_;
   };
}
#endif
