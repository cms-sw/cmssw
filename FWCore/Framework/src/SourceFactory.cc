// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     SourceFactory
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Wed May 25 19:27:37 EDT 2005
// $Id: SourceFactory.cc,v 1.2 2005/05/26 16:37:45 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/CoreFramework/interface/SourceFactory.h"
#include "FWCore/CoreFramework/interface/EventSetupProvider.h"


//
// static member functions
//
namespace edm {
   namespace eventsetup {
      std::string SourceMakerTraits::name() { return "EventSetupSourceFactory"; }
      
   }
}

COMPONENTFACTORY_GET( edm::eventsetup::SourceMakerTraits )
