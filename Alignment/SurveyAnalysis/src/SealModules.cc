#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "Alignment/SurveyAnalysis/interface/SurveyAlignmentAlgorithm.h"
#include "Alignment/SurveyAnalysis/interface/SurveyDBUploader.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN(AlignmentAlgorithmPluginFactory, SurveyAlignmentAlgorithm, "SurveyAlignmentAlgorithm");
DEFINE_ANOTHER_FWK_MODULE(SurveyDBUploader);
