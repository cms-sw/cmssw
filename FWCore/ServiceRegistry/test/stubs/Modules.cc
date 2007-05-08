/*
 *  Modules.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 9/7/05.
 *
 */

#include "FWCore/ServiceRegistry/test/stubs/DependsOnDummyService.h"
#include "FWCore/ServiceRegistry/test/stubs/DummyService.h"
#include "FWCore/ServiceRegistry/test/stubs/DummyServiceE0.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using namespace testserviceregistry;
using namespace edm::serviceregistry;
DEFINE_FWK_SERVICE_MAKER(DependsOnDummyService,NoArgsMaker<DependsOnDummyService>);
DEFINE_FWK_SERVICE(DummyService);
DEFINE_FWK_SERVICE(DummyServiceE0);
DEFINE_FWK_SERVICE(DummyServiceA1);
DEFINE_FWK_SERVICE(DummyServiceD2);
DEFINE_FWK_SERVICE(DummyServiceB3);
DEFINE_FWK_SERVICE(DummyServiceC4);
