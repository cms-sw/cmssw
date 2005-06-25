#ifndef EVENTSETUP_MODULEFACTORY_H
#define EVENTSETUP_MODULEFACTORY_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     ModuleFactory
// 
/**\class ModuleFactory ModuleFactory.h Core/CoreFramework/interface/ModuleFactory.h

 Description: Factory which is dynamically loadable and used to create an eventstore module

 Usage:
    Used by the SEAL plugin-manager

*/
//
// Author:      Chris Jones
// Created:     Wed May 25 18:01:31 EDT 2005
// $Id: ModuleFactory.h,v 1.3 2005/06/23 19:59:30 wmtan Exp $
//

// system include files
#include <string>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/CoreFramework/interface/ComponentFactory.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      class DataProxyProvider;

      struct ModuleMakerTraits {
         static std::string name();
         static void addTo(EventSetupProvider& iProvider, boost::shared_ptr<DataProxyProvider> iComponent) ;
      };
      template< class TType>
         struct ModuleMaker : public ComponentMaker<edm::eventsetup::ModuleMakerTraits,TType> {};
      
      typedef  ComponentFactory<ModuleMakerTraits> ModuleFactory ;
   }
}

#define DEFINE_FWK_EVENTSETUP_MODULE(type) \
DEFINE_SEAL_MODULE (); \
DEFINE_SEAL_PLUGIN (edm::eventsetup::ModuleFactory,edm::eventsetup::ModuleMaker<type>,#type);

#define DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(type) \
DEFINE_SEAL_PLUGIN (edm::eventsetup::ModuleFactory,edm::eventsetup::ModuleMaker<type>,#type);

#endif /* EVENTSETUP_MODULEFACTORY_H */

