#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "QCDAnalysis/UEAnalysis/interface/AnalysisRootpleProducer.h"
#include "QCDAnalysis/UEAnalysis/interface/AnalysisRootpleProducerOnlyMC.h"
#include "QCDAnalysis/UEAnalysis/interface/UEJetValidation.h"
#include "QCDAnalysis/UEAnalysis/interface/UEJetMultiplicity.h"

DEFINE_FWK_MODULE(AnalysisRootpleProducer);
DEFINE_FWK_MODULE(AnalysisRootpleProducerOnlyMC);
DEFINE_FWK_MODULE(UEJetValidation);
DEFINE_FWK_MODULE(UEJetMultiplicity);
