#ifndef FWCore_Framework_MakeModuleHelper_h
#define FWCore_Framework_MakeModuleHelper_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     MakeModuleHelper
//
/**\class edm::MakeModuleHelper MakeModuleHelper.h "MakeModuleHelper.h"

 Description: A template class which can be specialized to create a module from a user type

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun, 25 Aug 2013 20:54:45 GMT
//

// system include files
#include <memory>
// user include files

// forward declarations
namespace edm {
  class ParameterSet;

  template <typename Base>
  class MakeModuleHelper {
  public:
    MakeModuleHelper() = delete;
    MakeModuleHelper(const MakeModuleHelper&) = delete;  // stop default

    const MakeModuleHelper& operator=(const MakeModuleHelper&) = delete;  // stop default

    template <typename T>
    static std::unique_ptr<Base> makeModule(ParameterSet const& pset) {
      auto module = std::make_unique<T>(pset);
      return std::unique_ptr<Base>(module.release());
    }
  };
}  // namespace edm

#endif
