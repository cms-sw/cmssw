#include "PluginManager/ModuleDef.h"
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
#include "OnlineDB/CSCCondDB/interface/CSCChamberIndexHandler.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberMapHandler.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberMapPopConAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberIndexPopConAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCCrateMapHandler.h"
#include "OnlineDB/CSCCondDB/interface/CSCCrateMapPopConAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCDDUMapHandler.h"
#include "OnlineDB/CSCCondDB/interface/CSCDDUMapPopConAnalyzer.h"
#include "OnlineDB/CSCCondDB/interface/CSCCrateMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCDDUMapValues.h"
#include "OnlineDB/CSCCondDB/interface/WriteChamberMapValuesToDB.h"
#include "OnlineDB/CSCCondDB/interface/WriteCrateMapValuesToDB.h"
#include "OnlineDB/CSCCondDB/interface/WriteChamberIndexValuesToDB.h"
#include "OnlineDB/CSCCondDB/interface/WriteDDUMapValuesToDB.h"

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
DEFINE_ANOTHER_FWK_MODULE(WriteChamberMapValuesToDB);
DEFINE_ANOTHER_FWK_MODULE(WriteCrateMapValuesToDB);
DEFINE_ANOTHER_FWK_MODULE(WriteChamberIndexValuesToDB);
DEFINE_ANOTHER_FWK_MODULE(WriteDDUMapValuesToDB);
DEFINE_ANOTHER_FWK_MODULE(CSCChamberIndexPopConAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCChamberMapPopConAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCCrateMapPopConAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCDDUMapPopConAnalyzer);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCChamberMapValues);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCChamberIndexValues);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCCrateMapValues);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCDDUMapValues);
