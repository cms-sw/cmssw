// -*- C++ -*-
//
// Package:     Framework
// Class  :     ModuleFactory
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Wed May 25 19:27:44 EDT 2005
// $Id: ModuleFactory.cc,v 1.5 2007/04/13 10:39:42 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"

//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {

//
// static member functions
//
       std::string ModuleMakerTraits::name() { return "CMS EDM Framework ESModule"; }
      void ModuleMakerTraits::addTo(EventSetupProvider& iProvider, boost::shared_ptr<DataProxyProvider> iComponent) 
      {
         iProvider.add(iComponent);
      }

      boost::shared_ptr<ModuleMakerTraits::base_type> const*
      ModuleMakerTraits::getAlreadyMadeComponent(EventSetupsController const&,
                                                 ParameterSet const&) {
         return 0;
      }

      void ModuleMakerTraits::putComponent(EventSetupsController&,
                                           ParameterSet const&,
                                           boost::shared_ptr<base_type> const&) {
      }
   }
}
COMPONENTFACTORY_GET(edm::eventsetup::ModuleMakerTraits);
