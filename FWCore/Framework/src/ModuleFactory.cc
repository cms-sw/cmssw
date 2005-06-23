// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     ModuleFactory
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Wed May 25 19:27:44 EDT 2005
// $Id: ModuleFactory.cc,v 1.1 2005/05/29 02:29:54 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/CoreFramework/interface/ModuleFactory.h"
#include "FWCore/CoreFramework/interface/EventSetupProvider.h"

//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {

//
// static member functions
//
       std::string ModuleMakerTraits::name() { return "EventSetupModuleFactory"; }
      void ModuleMakerTraits::addTo(EventSetupProvider& iProvider, boost::shared_ptr<DataProxyProvider> iComponent) 
      {
         iProvider.add(iComponent);
      }
   }
}
COMPONENTFACTORY_GET(edm::eventsetup::ModuleMakerTraits)
