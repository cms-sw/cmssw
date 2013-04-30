#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQM/DataScouting/plugins/RazorVarAnalyzer.h"
#include "DQM/DataScouting/plugins/AlphaTVarAnalyzer.h"
#include "DQM/DataScouting/plugins/DiJetVarAnalyzer.h"
#include "DQM/DataScouting/plugins/ScoutingTestAnalyzer.h"

DEFINE_FWK_MODULE(RazorVarAnalyzer);
DEFINE_FWK_MODULE(AlphaTVarAnalyzer);
DEFINE_FWK_MODULE(DiJetVarAnalyzer);
DEFINE_FWK_MODULE(ScoutingTestAnalyzer);
