// -*- C++ -*-
//
// Package:     Framework
// Class  :     LooperFactory
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Wed May 25 19:27:37 EDT 2005
// $Id: LooperFactory.cc,v 1.5 2012/06/06 15:51:21 wdd Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/LooperFactory.h"

//
// static member functions
//
namespace edm {
   namespace eventsetup {
      std::string LooperMakerTraits::name() { return "CMS EDM Framework EDLooper"; }
      
      void 
      LooperMakerTraits::replaceExisting(EventSetupProvider& iProvider, boost::shared_ptr<EDLooperBase> iComponent) {
         throw edm::Exception(edm::errors::LogicError)
            << "LooperMakerTraits::replaceExisting\n"
            << "This function is not implemented and should never be called.\n"
            << "Please report this to a Framework Developer\n";
      }

      boost::shared_ptr<LooperMakerTraits::base_type>
      LooperMakerTraits::getComponentAndRegisterProcess(EventSetupsController&,
                                                        ParameterSet const&) {
        return boost::shared_ptr<LooperMakerTraits::base_type>();
      }

      void LooperMakerTraits::putComponent(EventSetupsController&,
                                           ParameterSet const&,
                                           boost::shared_ptr<base_type> const&) {
      }
   }
}

COMPONENTFACTORY_GET(edm::eventsetup::LooperMakerTraits);
