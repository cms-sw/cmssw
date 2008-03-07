
#include "FWCore/Framework/interface/MakerMacros.h"
#include "AnalysisExamples/SiStripDetectorPerformance/plugins/ClusterAnalysis.h"
#include "AnalysisExamples/SiStripDetectorPerformance/plugins/ClusterAnalysisFilter.h"
<<<<<<< SealModules.cc
//#include "AnalysisExamples/SiStripDetectorPerformance/plugins/ClusterThr.h"
=======
>>>>>>> 1.4

using cms::ClusterAnalysis;
using cms::ClusterAnalysisFilter;
<<<<<<< SealModules.cc
//using cms::ClusterThr;
=======
>>>>>>> 1.4

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ClusterAnalysis);
DEFINE_ANOTHER_FWK_MODULE(ClusterAnalysisFilter);
<<<<<<< SealModules.cc
//DEFINE_ANOTHER_FWK_MODULE(ClusterThr);
=======
>>>>>>> 1.4
