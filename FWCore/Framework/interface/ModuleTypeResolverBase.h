#ifndef FWCore_Framework_ModuleTypeResolverBase_h
#define FWCore_Framework_ModuleTypeResolverBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ModuleTypeResolverBase
//
/**\class edm::ModuleTypeResolverBase ModuleTypeResolverBase.h "FWCore/Framework/interface/ModuleTypeResolverBase.h"

 Description: Base class for deriving alternative module types to use when loading

 Usage:
        This is meant to be used as part of a do...while loop. The condition should be the returned int is not kLastIndex and the
 type returned is not what you need.
*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 27 Apr 2022 16:21:10 GMT
//

// system include files
#include <string>

// user include files

// forward declarations
namespace edm {
  class ModuleTypeResolverBase {
  public:
    static constexpr int kInitialIndex = 0;
    static constexpr int kLastIndex = -1;
    virtual ~ModuleTypeResolverBase() = default;

    /**This function is meant to be called multiple times with different values for index. The first call should set index
       to kInitialIndex. The int returned from the function is the new index to use on next call or is a value of kLastIndex which
       means no further calls should be made. The returned string is the next concrete type to be used when making a call.
       On subsequent call, the argument basename can be the same string as returned from the previous call to the function.
        **/
    virtual std::pair<std::string, int> resolveType(std::string basename, int index) const = 0;
  };
}  // namespace edm

#endif
