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
// $Id: ModuleFactory.cc,v 1.6 2012/04/16 15:43:50 wdd Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/src/EventSetupsController.h"

//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {

//
// static member functions
//
      std::string ModuleMakerTraits::name() { return "CMS EDM Framework ESModule"; }
      void ModuleMakerTraits::addTo(EventSetupProvider& iProvider,
                                    boost::shared_ptr<DataProxyProvider> iComponent,
                                    ParameterSet const&,
                                    bool) 
      {
         iProvider.add(iComponent);
      }

      void ModuleMakerTraits::replaceExisting(EventSetupProvider& iProvider, boost::shared_ptr<DataProxyProvider> iComponent) 
      {
         iProvider.replaceExisting(iComponent);
      }

      boost::shared_ptr<ModuleMakerTraits::base_type>
      ModuleMakerTraits::getComponentAndRegisterProcess(EventSetupsController& esController,
                                                        ParameterSet const& iConfiguration) {
         return esController.getESProducerAndRegisterProcess(iConfiguration, esController.indexOfNextProcess());
      }

      void ModuleMakerTraits::putComponent(EventSetupsController& esController,
                                           ParameterSet const& iConfiguration,
                                           boost::shared_ptr<base_type> const& component) {
         esController.putESProducer(iConfiguration, component, esController.indexOfNextProcess());
      }
   }
}
COMPONENTFACTORY_GET(edm::eventsetup::ModuleMakerTraits);
