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
// $Id: SourceFactory.cc,v 1.8 2012/06/06 15:51:21 wdd Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/src/EventSetupsController.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// static member functions
//
namespace edm {
   namespace eventsetup {

      std::string SourceMakerTraits::name() { return "CMS EDM Framework ESSource"; }

      void 
      SourceMakerTraits::replaceExisting(EventSetupProvider& iProvider, boost::shared_ptr<EventSetupRecordIntervalFinder> iComponent) {
         throw edm::Exception(edm::errors::LogicError)
            << "SourceMakerTraits::replaceExisting\n"
            << "This function is not implemented and should never be called.\n"
            << "Please report this to a Framework Developer\n";
      }

      boost::shared_ptr<SourceMakerTraits::base_type>
      SourceMakerTraits::getComponentAndRegisterProcess(EventSetupsController& esController,
                                                        ParameterSet const& iConfiguration) {
         return esController.getESSourceAndRegisterProcess(iConfiguration, esController.indexOfNextProcess());
      }

      void SourceMakerTraits::putComponent(EventSetupsController& esController,
                                           ParameterSet const& iConfiguration,
                                           boost::shared_ptr<base_type> const& component) {
         esController.putESSource(iConfiguration, component, esController.indexOfNextProcess());
      }

      void SourceMakerTraits::logInfoWhenSharing(ParameterSet const& iConfiguration) {

         std::string edmtype = iConfiguration.getParameter<std::string>("@module_edm_type");
         std::string modtype = iConfiguration.getParameter<std::string>("@module_type");
         std::string label = iConfiguration.getParameter<std::string>("@module_label");
         edm::LogVerbatim("EventSetupSharing") << "Sharing " << edmtype << ": class=" << modtype << " label='" << label << "'";
      }
   }
}

COMPONENTFACTORY_GET(edm::eventsetup::SourceMakerTraits);
