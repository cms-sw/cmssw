#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Calibration/HcalCalibAlgos/interface/Analyzer_minbias.h"
#include "Calibration/HcalCalibAlgos/interface/HcalConstantsASCIIWriter.h"
#include "Calibration/HcalCalibAlgos/src/HitReCalibrator.h"

#include "Calibration/HcalCalibAlgos/interface/HcalCalibrator.h"

using cms::Analyzer_minbias;
using cms::HcalConstantsASCIIWriter;
using cms::HitReCalibrator;

DEFINE_FWK_MODULE(Analyzer_minbias);
DEFINE_FWK_MODULE(HcalConstantsASCIIWriter);
DEFINE_FWK_MODULE(HitReCalibrator);


