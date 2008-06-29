#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "OnlineDB/CSCCondDB/interface/CSCAFEBAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCCompThreshAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCCrossTalkAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCOldCrossTalkAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCGainAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCNoiseMatrixAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCscaAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCCFEBConnectivityAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCSaturationAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberIndexValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCCrateMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCDDUMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCMap1Read.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CSCAFEBAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCCompThreshAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCCrossTalkAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCOldCrossTalkAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCGainAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCNoiseMatrixAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCscaAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCCFEBConnectivityAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCSaturationAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCMap1Read);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCChamberMapValues);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCChamberIndexValues);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCCrateMapValues);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCDDUMapValues);
