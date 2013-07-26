// -*- C++ -*-
//
// Package:     Package
// Class  :     getAllTriggerNames
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Fri, 26 Jul 2013 20:43:45 GMT
// $Id$
//

// system include files

// user include files
#include "FWCore/Framework/interface/getAllTriggerNames.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"


namespace edm {
  
  std::vector<std::string> const& getAllTriggerNames() {
    Service<service::TriggerNamesService> tns;
    return tns->getTrigPaths();
  }
}
