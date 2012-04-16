// -*- C++ -*-
//
// Package:     Framework
// Class  :     SourceFactory
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Wed May 25 19:27:37 EDT 2005
// $Id: SourceFactory.cc,v 1.6 2007/06/29 03:43:22 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/src/EventSetupsController.h"

//
// static member functions
//
namespace edm {
   namespace eventsetup {

      std::string SourceMakerTraits::name() { return "CMS EDM Framework ESSource"; }

      boost::shared_ptr<SourceMakerTraits::base_type> const*
      SourceMakerTraits::getAlreadyMadeComponent(EventSetupsController const& esController,
                                                 ParameterSet const& iConfiguration) {
         return esController.getAlreadyMadeESSource(iConfiguration);
      }

      void SourceMakerTraits::putComponent(EventSetupsController& esController,
                                           ParameterSet const& iConfiguration,
                                           boost::shared_ptr<base_type> const& component) {
         esController.putESSource(iConfiguration, component);
      }
   }
}

COMPONENTFACTORY_GET(edm::eventsetup::SourceMakerTraits);
