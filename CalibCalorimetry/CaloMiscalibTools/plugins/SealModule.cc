#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/EcalRecHitRecalib.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibTools.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibToolsMC.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/HcalRecHitRecalib.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/WriteEcalMiscalibConstants.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/WriteEcalMiscalibConstantsMC.h"


DEFINE_FWK_EVENTSETUP_SOURCE(CaloMiscalibTools);
DEFINE_FWK_EVENTSETUP_SOURCE(CaloMiscalibToolsMC);
DEFINE_FWK_MODULE(EcalRecHitRecalib);
DEFINE_FWK_MODULE(HcalRecHitRecalib);
DEFINE_FWK_MODULE(WriteEcalMiscalibConstants);
DEFINE_FWK_MODULE(WriteEcalMiscalibConstantsMC);

