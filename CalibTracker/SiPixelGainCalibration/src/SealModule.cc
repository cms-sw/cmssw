#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

//define this as a plug-in
// include header file here:
#include "CalibTracker/SiPixelGainCalibration/interface/SiPixelGainCalibrationAnalysis.h"
DEFINE_ANOTHER_FWK_MODULE(SiPixelGainCalibrationAnalysis);

//define this as a plug-in
// include header file here:
#include "CalibTracker/SiPixelGainCalibration/interface/SiPixelGainCalibrationUnpackLocal.h"
DEFINE_ANOTHER_FWK_MODULE(SiPixelGainCalibrationUnpackLocal);

//define this as a plug-in
// include header file here:
#include "CalibTracker/SiPixelGainCalibration/interface/SiPixelGainCalibrationDBAnalysis.h"
DEFINE_ANOTHER_FWK_MODULE(SiPixelGainCalibrationDBAnalysis);

