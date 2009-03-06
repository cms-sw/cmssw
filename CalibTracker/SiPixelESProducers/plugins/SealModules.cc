//#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeGainESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeGainForHLTESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeGainOfflineESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeLorentzAngleESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileWriter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiPixelFakeGainESSource);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiPixelFakeGainForHLTESSource);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiPixelFakeGainOfflineESSource);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiPixelFakeLorentzAngleESSource);
DEFINE_ANOTHER_FWK_MODULE(SiPixelDetInfoFileWriter);
