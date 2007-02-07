#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/AnalyzeTracksClusters.h"
//#include "AnalysisExamples/SiStripDetectorPerformance/interface/ClusterAnalysis.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/ClusterAnalysisFilter.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/MTCCNtupleMaker.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/CosmicTIFFilter.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/CosmicGenFilter.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackerMuFilter.h"
<<<<<<< SealModules.cc
// M. De Mattia 25/1/2007
#include "AnalysisExamples/SiStripDetectorPerformance/interface/TIFNtupleMaker.h"
// M. De Mattia 30/1/2007
#include "AnalysisAlgos/SiStripClusterInfoProducer/interface/SiStripFakeRawDigiModule.h"
=======
#include "AnalysisExamples/SiStripDetectorPerformance/interface/MTCCAmplifyDigis.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/AnaSiStripClusters.h"
>>>>>>> 1.7

//using cms::ClusterAnalysis;
using cms::ClusterAnalysisFilter;
using cms::CosmicTIFFilter;
using cms::CosmicGenFilter;
using cms::TrackerMuFilter;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(AnalyzeTracksClusters);
//DEFINE_ANOTHER_FWK_MODULE(ClusterAnalysis);
DEFINE_ANOTHER_FWK_MODULE(ClusterAnalysisFilter);
DEFINE_ANOTHER_FWK_MODULE(MTCCNtupleMaker);
DEFINE_ANOTHER_FWK_MODULE(CosmicTIFFilter);
DEFINE_ANOTHER_FWK_MODULE(CosmicGenFilter);
DEFINE_ANOTHER_FWK_MODULE(TrackerMuFilter);
<<<<<<< SealModules.cc
// 25/1/2007
DEFINE_ANOTHER_FWK_MODULE(TIFNtupleMaker);
// 25/1/2007
DEFINE_ANOTHER_FWK_MODULE(SiStripFakeRawDigiModule);
=======
DEFINE_ANOTHER_FWK_MODULE(MTCCAmplifyDigis);
DEFINE_ANOTHER_FWK_MODULE(AnaSiStripClusters);
>>>>>>> 1.7
