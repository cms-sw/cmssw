// -*- C++ -*-
//
// Package:     test
// Class  :     DummyStoreConfigService
// 
/**\class DummyStoreConfigService DummyStoreConfigService.h ServiceRegistry/test/interface/DummyStoreConfigService.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Mar 12 15:41:10 CST 2010
//

// system include files

// user include files
#include "FWCore/ServiceRegistry/interface/SaveConfiguration.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

// forward declarations

class DummyStoreConfigService : public edm::serviceregistry::SaveConfiguration
{

   public:
   DummyStoreConfigService() {}
};

DEFINE_FWK_SERVICE_MAKER(DummyStoreConfigService,edm::serviceregistry::NoArgsMaker<DummyStoreConfigService>);
