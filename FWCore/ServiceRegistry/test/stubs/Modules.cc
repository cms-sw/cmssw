/*
 *  Modules.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 9/7/05.
 *
 */

#include "FWCore/Utilities/interface/concatenate.h"
#include "FWCore/ServiceRegistry/test/stubs/DependsOnDummyService.h"
#include "FWCore/ServiceRegistry/test/stubs/DummyService.h"
#include "FWCore/ServiceRegistry/test/stubs/DummyServiceE0.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using namespace testserviceregistry;
using namespace edm::serviceregistry;
DEFINE_FWK_SERVICE_MAKER(DependsOnDummyService, NoArgsMaker<DependsOnDummyService>);
DEFINE_FWK_SERVICE(DummyService);
DEFINE_FWK_SERVICE(EDM_CONCATENATE(DummyService, E0));
DEFINE_FWK_SERVICE(EDM_CONCATENATE(DummyService, A1));
DEFINE_FWK_SERVICE(EDM_CONCATENATE(DummyService, D2));
DEFINE_FWK_SERVICE(EDM_CONCATENATE(DummyService, B3));
DEFINE_FWK_SERVICE(EDM_CONCATENATE(DummyService, C4));
