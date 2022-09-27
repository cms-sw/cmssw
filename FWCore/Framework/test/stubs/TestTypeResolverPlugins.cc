#include "FWCore/Framework/test/TestTypeResolvers.h"

#include "FWCore/Framework/interface/ModuleTypeResolverMakerFactory.h"
DEFINE_EDM_PLUGIN(edm::ModuleTypeResolverMakerFactory,
                  edm::test::ConfigurableTestTypeResolverMaker,
                  "edm::test::ConfigurableTestTypeResolverMaker");
