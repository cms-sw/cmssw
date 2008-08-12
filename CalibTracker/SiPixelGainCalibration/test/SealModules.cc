#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CalibTracker/SiPixelGainCalibration/test/SimpleTestPrintOutPixelCalibAnalyzer.h"
//#include "CalibTracker/SiPixelGainCalibration/test/SiPixelGainCalibrationReadDQMFile.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(SimpleTestPrintOutPixelCalibAnalyzer);
//DEFINE_ANOTHER_FWK_MODULE(SiPixelGainCalibrationReadDQMFile);
