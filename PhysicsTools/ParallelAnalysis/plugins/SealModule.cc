#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/ParallelAnalysis/interface/TSelectorAnalyzer.h"
#include "PhysicsTools/ParallelAnalysis/interface/TrackAnalysisAlgorithm.h"

typedef TSelectorAnalyzer<examples::TrackAnalysisAlgorithm> TrackTSelectorAnalyzer;


DEFINE_FWK_MODULE( TrackTSelectorAnalyzer );
