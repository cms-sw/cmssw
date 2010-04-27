#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "QCDAnalysis/UEAnalysis/interface/AnalysisRootpleProducer.h"
#include "QCDAnalysis/UEAnalysis/interface/AnalysisRootpleProducerOnlyMC.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(AnalysisRootpleProducer);
DEFINE_ANOTHER_FWK_MODULE(AnalysisRootpleProducerOnlyMC);
