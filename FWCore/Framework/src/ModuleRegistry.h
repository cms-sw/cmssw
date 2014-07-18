#ifndef FWCore_Framework_ModuleRegistry_h
#define FWCore_Framework_ModuleRegistry_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ModuleRegistry
// 
/**\class edm::ModuleRegistry ModuleRegistry.h "FWCore/Framework/src/ModuleRegistry.h"

 Description: Constructs and owns framework modules

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 23 Aug 2013 16:21:10 GMT
//

// system include files
#include <map>
#include <memory>
#include <string>

// user include files
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

// forward declarations
namespace edm {
  class ParameterSet;
  struct MakeModuleParams;
  class ModuleDescription;
  class PreallocationConfiguration;
  namespace maker {
    class ModuleHolder;
  }
  
  class ModuleRegistry
  {
  public:
    std::shared_ptr<maker::ModuleHolder> getModule(MakeModuleParams const& p,
                                                   std::string const& moduleLabel,
                                                   signalslot::Signal<void(ModuleDescription const&)>& iPre,
                                                   signalslot::Signal<void(ModuleDescription const&)>& iPost);
    
    maker::ModuleHolder* replaceModule(std::string const& iModuleLabel,
                                       edm::ParameterSet const& iPSet,
                                       edm::PreallocationConfiguration const&);
    
    template <typename F>
    void forAllModuleHolders(F iFunc) {
      for(auto const& labelMod: labelToModule_) {
        maker::ModuleHolder* t = labelMod.second.get();
        iFunc(t);
      }
    }
  private:
    std::map<std::string,std::shared_ptr<maker::ModuleHolder>> labelToModule_;
  };
}


#endif
