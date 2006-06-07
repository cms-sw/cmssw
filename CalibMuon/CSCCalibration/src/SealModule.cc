#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CalibMuon/CSCCalibration/interface/CSCAFEBAnalyzer.h"
#include "CalibMuon/CSCCalibration/interface/CSCAFEBdacAnalyzer.h"
#include "CalibMuon/CSCCalibration/interface/CSCCompThreshAnalyzer.h"
#include "CalibMuon/CSCCalibration/interface/CSCCrossTalkAnalyzer.h"
#include "CalibMuon/CSCCalibration/interface/CSCGainAnalyzer.h"
#include "CalibMuon/CSCCalibration/interface/CSCNoiseMatrixAnalyzer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CSCAFEBAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCAFEBdacAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCCompThreshAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCCrossTalkAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCGainAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCNoiseMatrixAnalyzer);
