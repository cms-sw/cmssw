#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "CalibTracker/SiStripDCS/interface/SiStripDetVOffBuilder.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(SiStripDetVOffBuilder);

// EDFilter on the max number of modules with HV off
#include "CalibTracker/SiStripDCS/plugins/FilterTrackerOn.h"
DEFINE_FWK_MODULE(FilterTrackerOn);
