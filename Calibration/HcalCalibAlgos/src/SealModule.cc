#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Calibration/HcalCalibAlgos/interface/Analyzer_minbias.h"
#include "Calibration/HcalCalibAlgos/src/GammaJetAnalysis.h"
#include "Calibration/HcalCalibAlgos/interface/DiJetAnalyzer.h"
#include "Calibration/HcalCalibAlgos/interface/HcalConstantsASCIIWriter.h"
#include "Calibration/HcalCalibAlgos/src/HitReCalibrator.h"
using cms::Analyzer_minbias;
using cms::GammaJetAnalysis;
using cms::DiJetAnalyzer;
using cms::HcalConstantsASCIIWriter;
using cms::HitReCalibrator;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(Analyzer_minbias);
DEFINE_ANOTHER_FWK_MODULE(GammaJetAnalysis);
DEFINE_ANOTHER_FWK_MODULE(DiJetAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(HcalConstantsASCIIWriter);
DEFINE_ANOTHER_FWK_MODULE(HitReCalibrator);
