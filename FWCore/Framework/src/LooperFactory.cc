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
// $Id: LooperFactory.cc,v 1.2 2007/04/13 10:39:42 wmtan Exp $
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
      
   }
}

COMPONENTFACTORY_GET(edm::eventsetup::LooperMakerTraits);
