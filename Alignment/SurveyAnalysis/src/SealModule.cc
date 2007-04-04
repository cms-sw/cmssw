#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Alignment/SurveyAnalysis/interface/SurveyDataConverter.h"
#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentAlgorithm.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"

DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN(AlignmentAlgorithmPluginFactory, SurveyAlignmentAlgorithm, "SurveyAlignmentAlgorithm");
DEFINE_ANOTHER_FWK_MODULE(SurveyDataConverter);
