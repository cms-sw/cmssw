//#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
//#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeGainESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileWriter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiPixelFakeGainESSource);
DEFINE_ANOTHER_FWK_MODULE(SiPixelDetInfoFileWriter);
