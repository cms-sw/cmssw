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
// $Id: SourceFactory.cc,v 1.2 2005/06/23 19:59:48 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"


//
// static member functions
//
namespace edm {
   namespace eventsetup {
      std::string SourceMakerTraits::name() { return "EventSetupSourceFactory"; }
      
   }
}

COMPONENTFACTORY_GET(edm::eventsetup::SourceMakerTraits)
