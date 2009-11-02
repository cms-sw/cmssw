#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMOffline/PFTau/plugins/CandidateBenchmarkAnalyzer.h"
#include "DQMOffline/PFTau/plugins/PFCandidateBenchmarkAnalyzer.h"
#include "DQMOffline/PFTau/plugins/PFCandidateManagerAnalyzer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE (PFCandidateBenchmarkAnalyzer) ;
DEFINE_ANOTHER_FWK_MODULE (CandidateBenchmarkAnalyzer) ;
DEFINE_ANOTHER_FWK_MODULE (PFCandidateManagerAnalyzer) ;
