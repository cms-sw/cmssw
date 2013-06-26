//#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeGainESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeGainForHLTESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeGainOfflineESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeLorentzAngleESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileWriter.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeCPEGenericErrorParmESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeTemplateDBObjectESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelQualityESProducer.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeQualityESSource.h"


DEFINE_FWK_EVENTSETUP_SOURCE(SiPixelFakeGainESSource);
DEFINE_FWK_EVENTSETUP_SOURCE(SiPixelFakeGainForHLTESSource);
DEFINE_FWK_EVENTSETUP_SOURCE(SiPixelFakeGainOfflineESSource);
DEFINE_FWK_EVENTSETUP_SOURCE(SiPixelFakeLorentzAngleESSource);
DEFINE_FWK_EVENTSETUP_SOURCE(SiPixelFakeQualityESSource);
DEFINE_FWK_EVENTSETUP_SOURCE(SiPixelFakeCPEGenericErrorParmESSource);
DEFINE_FWK_EVENTSETUP_SOURCE(SiPixelFakeTemplateDBObjectESSource);
DEFINE_FWK_EVENTSETUP_MODULE(SiPixelQualityESProducer);
DEFINE_FWK_MODULE(SiPixelDetInfoFileWriter);
