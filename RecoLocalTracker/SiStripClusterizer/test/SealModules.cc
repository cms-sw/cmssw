#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

#include "RecoLocalTracker/SiStripClusterizer/test/CompareClusters.h"
#include "RecoLocalTracker/SiStripClusterizer/test/ClusterizerUnitTester.h"
#include "RecoLocalTracker/SiStripClusterizer/test/StripByStripTestDriver.h"
#include "RecoLocalTracker/SiStripClusterizer/test/ClusterizerUnitTesterESProducer.h"


DEFINE_FWK_MODULE(CompareClusters);
DEFINE_FWK_MODULE(ClusterizerUnitTester);
DEFINE_FWK_MODULE(StripByStripTestDriver);
DEFINE_FWK_EVENTSETUP_MODULE(ClusterizerUnitTesterESProducer);

