#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/AnalyzeTracksClusters.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/ClusterAnalysis.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/ClusterAnalysisFilter.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/MTCCNtupleMaker.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/CosmicTIFFilter.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/CosmicGenFilter.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackerMuFilter.h"
// M. De Mattia 25/1/2007
#include "AnalysisExamples/SiStripDetectorPerformance/interface/TIFNtupleMaker.h"
using cms::ClusterAnalysis;
using cms::ClusterAnalysisFilter;
using cms::CosmicTIFFilter;
using cms::CosmicGenFilter;
using cms::TrackerMuFilter;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(AnalyzeTracksClusters);
DEFINE_ANOTHER_FWK_MODULE(ClusterAnalysis);
DEFINE_ANOTHER_FWK_MODULE(ClusterAnalysisFilter);
DEFINE_ANOTHER_FWK_MODULE(MTCCNtupleMaker);
DEFINE_ANOTHER_FWK_MODULE(CosmicTIFFilter);
DEFINE_ANOTHER_FWK_MODULE(CosmicGenFilter);
DEFINE_ANOTHER_FWK_MODULE(TrackerMuFilter);
// 25/1/2007
DEFINE_ANOTHER_FWK_MODULE(TIFNtupleMaker);

