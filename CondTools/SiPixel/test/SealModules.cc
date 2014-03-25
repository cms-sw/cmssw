#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondTools/SiPixel/test/SiPixelCondObjBuilder.h"
#include "CondTools/SiPixel/test/SiPixelCondObjReader.h"
#include "CondTools/SiPixel/test/SiPixelCondObjForHLTBuilder.h"
#include "CondTools/SiPixel/test/SiPixelCondObjForHLTReader.h"
#include "CondTools/SiPixel/test/SiPixelCondObjOfflineBuilder.h"
#include "CondTools/SiPixel/test/SiPixelCondObjOfflineReader.h"
#include "CondTools/SiPixel/test/SiPixelCondObjAllPayloadsReader.h"

#include "CondTools/SiPixel/test/SiPixelGainCalibrationReadDQMFile.h"
#include "CondTools/SiPixel/test/SiPixelGainCalibrationRejectNoisyAndDead.h"

#include "CondTools/SiPixel/test/SiPixelBadModuleByHandBuilder.h"
#include "CondTools/SiPixel/test/SiPixelBadModuleReader.h"

#include "CondTools/SiPixel/test/SiPixelLorentzAngleReader.h"
#include "CondTools/SiPixel/test/SiPixelLorentzAngleDB.h"

#include "CondTools/SiPixel/test/SiPixelPerformanceSummaryBuilder.h"
#include "CondTools/SiPixel/test/SiPixelPerformanceSummaryReader.h"

#include "CondTools/SiPixel/test/SiPixelCPEGenericErrorParmReader.h"
#include "CondTools/SiPixel/test/SiPixelCPEGenericErrorParmUploader.h"
#include "CondTools/SiPixel/test/SiPixelFakeCPEGenericErrorParmSourceReader.h"

#include "CondTools/SiPixel/test/SiPixelTemplateDBObjectReader.h"
#include "CondTools/SiPixel/test/SiPixelTemplateDBObjectUploader.h"
#include "CondTools/SiPixel/test/SiPixelFakeTemplateDBSourceReader.h"

#include "CondFormats/DataRecord/interface/PixelDCSRcds.h"
#include "CondFormats/SiPixelObjects/interface/PixelDCSObject.h"
#include "CondTools/SiPixel/test/PixelDCSObjectReader.h"

using cms::SiPixelCondObjBuilder;
using cms::SiPixelCondObjReader;
using cms::SiPixelCondObjForHLTBuilder;
using cms::SiPixelCondObjForHLTReader;
using cms::SiPixelCondObjOfflineBuilder;
using cms::SiPixelCondObjOfflineReader;
using cms::SiPixelCondObjAllPayloadsReader;
using cms::SiPixelPerformanceSummaryBuilder;
using cms::SiPixelPerformanceSummaryReader;


DEFINE_FWK_MODULE(SiPixelCondObjBuilder);
DEFINE_FWK_MODULE(SiPixelCondObjReader);
DEFINE_FWK_MODULE(SiPixelCondObjForHLTBuilder);
DEFINE_FWK_MODULE(SiPixelCondObjForHLTReader);
DEFINE_FWK_MODULE(SiPixelCondObjOfflineBuilder);
DEFINE_FWK_MODULE(SiPixelCondObjOfflineReader);
DEFINE_FWK_MODULE(SiPixelCondObjAllPayloadsReader);
DEFINE_FWK_MODULE(SiPixelLorentzAngleReader);
DEFINE_FWK_MODULE(SiPixelLorentzAngleDB);
DEFINE_FWK_MODULE(SiPixelPerformanceSummaryBuilder);
DEFINE_FWK_MODULE(SiPixelPerformanceSummaryReader);
DEFINE_FWK_MODULE(SiPixelBadModuleByHandBuilder);
DEFINE_FWK_MODULE(SiPixelBadModuleReader);
DEFINE_FWK_MODULE(SiPixelGainCalibrationReadDQMFile);
DEFINE_FWK_MODULE(SiPixelGainCalibrationRejectNoisyAndDead);
DEFINE_FWK_MODULE(SiPixelCPEGenericErrorParmReader);
DEFINE_FWK_MODULE(SiPixelCPEGenericErrorParmUploader);
DEFINE_FWK_MODULE(SiPixelFakeCPEGenericErrorParmSourceReader);
DEFINE_FWK_MODULE(SiPixelTemplateDBObjectReader);
DEFINE_FWK_MODULE(SiPixelTemplateDBObjectUploader);
DEFINE_FWK_MODULE(SiPixelFakeTemplateDBSourceReader);
DEFINE_FWK_MODULE(PixelDCSObjectReader<PixelCaenChannelRcd>);
