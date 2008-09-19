#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/Framework/interface/ModuleFactory.h"
//#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "CalibTracker/SiStripCommon/plugins/SiStripDetInfoFileWriter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripDetInfoFileWriter);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripDetInfoFileReader);

#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
DEFINE_ANOTHER_FWK_SERVICE(TkDetMap);

