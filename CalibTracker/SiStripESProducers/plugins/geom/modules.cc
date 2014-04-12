#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"



#include "CalibTracker/SiStripESProducers/plugins/geom/SiStripHashedDetIdESModule.h"
DEFINE_FWK_EVENTSETUP_MODULE(SiStripHashedDetIdESModule);

#include "CalibTracker/SiStripESProducers/plugins/geom/SiStripConnectivity.h"
DEFINE_FWK_EVENTSETUP_MODULE(SiStripConnectivity);

#include "CalibTracker/SiStripESProducers/plugins/geom/SiStripRegionConnectivity.h"
DEFINE_FWK_EVENTSETUP_MODULE(SiStripRegionConnectivity);

