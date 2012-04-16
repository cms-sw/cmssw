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
// $Id: LooperFactory.cc,v 1.3 2007/06/29 03:43:21 wmtan Exp $
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
      
      boost::shared_ptr<LooperMakerTraits::base_type> const*
      LooperMakerTraits::getAlreadyMadeComponent(EventSetupsController const&,
                                                 ParameterSet const&) {
         return 0;
      }

      void LooperMakerTraits::putComponent(EventSetupsController&,
                                           ParameterSet const&,
                                           boost::shared_ptr<base_type> const&) {
      }
   }
}

COMPONENTFACTORY_GET(edm::eventsetup::LooperMakerTraits);
