#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "OnlineDB/CSCCondDB/interface/CSCAFEBAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCCompThreshAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCCrossTalkAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCOldCrossTalkAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCGainAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCOldGainAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCNoiseMatrixAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCscaAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCCFEBConnectivityAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCSaturationAnalyzer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CSCAFEBAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCCompThreshAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCCrossTalkAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCOldCrossTalkAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCGainAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCOldGainAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCNoiseMatrixAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCscaAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCCFEBConnectivityAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCSaturationAnalyzer);
