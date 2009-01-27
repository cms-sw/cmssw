
#include "FWCore/Framework/interface/MakerMacros.h"
#include "AnalysisExamples/SiStripDetectorPerformance/plugins/ClusterAnalysis.h"
#include "AnalysisExamples/SiStripDetectorPerformance/plugins/ClusterThr.h"

using cms::ClusterAnalysis;
using cms::ClusterThr;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ClusterAnalysis);
DEFINE_ANOTHER_FWK_MODULE(ClusterThr);
