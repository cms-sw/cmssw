#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DumpDBToFile.h"
#include "DumpFileToDB.h"
#include "DTT0Analyzer.h"
#include "DTTTrigAnalyzer.h"
#include "DTVDriftAnalyzer.h"
#include "ShiftTTrigDB.h"
#include "FakeTTrig.h"


DEFINE_FWK_MODULE(DumpDBToFile);
DEFINE_FWK_MODULE(DumpFileToDB);
DEFINE_FWK_MODULE(DTT0Analyzer);
DEFINE_FWK_MODULE(DTTTrigAnalyzer);
DEFINE_FWK_MODULE(DTVDriftAnalyzer);
DEFINE_FWK_MODULE(ShiftTTrigDB);
DEFINE_FWK_MODULE(FakeTTrig);

