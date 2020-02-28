#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "DQMOffline/PFTau/plugins/CandidateBenchmarkAnalyzer.h"
#include "DQMOffline/PFTau/plugins/METBenchmarkAnalyzer.h"
#include "DQMOffline/PFTau/plugins/MatchMETBenchmarkAnalyzer.h"
#include "DQMOffline/PFTau/plugins/PFCandidateBenchmarkAnalyzer.h"
#include "DQMOffline/PFTau/plugins/PFCandidateManagerAnalyzer.h"

DEFINE_FWK_MODULE(PFCandidateBenchmarkAnalyzer);
DEFINE_FWK_MODULE(CandidateBenchmarkAnalyzer);
DEFINE_FWK_MODULE(PFCandidateManagerAnalyzer);
DEFINE_FWK_MODULE(METBenchmarkAnalyzer);
DEFINE_FWK_MODULE(MatchMETBenchmarkAnalyzer);
