#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Calibration/HcalCalibAlgos/interface/Analyzer_minbias.h"
#include "Calibration/HcalCalibAlgos/src/GammaJetAnalysis.h"
#include "Calibration/HcalCalibAlgos/interface/DiJetAnalyzer.h"
#include "Calibration/HcalCalibAlgos/interface/HcalConstantsASCIIWriter.h"
#include "Calibration/HcalCalibAlgos/src/HitReCalibrator.h"

#include "Calibration/HcalCalibAlgos/interface/HcalCalibrator.h"

using cms::Analyzer_minbias;
using cms::GammaJetAnalysis;
using cms::DiJetAnalyzer;
using cms::HcalConstantsASCIIWriter;
using cms::HitReCalibrator;

//using cms::HcalCalibrator;


DEFINE_FWK_MODULE(Analyzer_minbias);
DEFINE_FWK_MODULE(GammaJetAnalysis);
DEFINE_FWK_MODULE(DiJetAnalyzer);
DEFINE_FWK_MODULE(HcalConstantsASCIIWriter);
DEFINE_FWK_MODULE(HitReCalibrator);

//DEFINE_FWK_MODULE(HcalCalibrator);

