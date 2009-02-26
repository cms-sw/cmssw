#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CalibCalorimetry/EcalLaserSorting/interface/WatcherSource.h"
#include "CalibCalorimetry/EcalLaserSorting/interface/LaserSorter.h"
#include "CalibCalorimetry/EcalLaserSorting/interface/LmfSource.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(WatcherSource);
DEFINE_ANOTHER_FWK_MODULE(LaserSorter);
DEFINE_ANOTHER_FWK_INPUT_SOURCE(LmfSource);
