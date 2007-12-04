#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/EcalRecHitRecalib.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibTools.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/HcalRecHitRecalib.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/WriteEcalMiscalibConstants.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/WriteHcalGains.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/WriteHcalPedestals.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/WriteHcalPedestalWidths.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/WriteHcalQIEData.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/WriteHcalElectronicsMap.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CaloMiscalibTools);
DEFINE_ANOTHER_FWK_MODULE(EcalRecHitRecalib);
DEFINE_ANOTHER_FWK_MODULE(HcalRecHitRecalib);
DEFINE_ANOTHER_FWK_MODULE(WriteEcalMiscalibConstants);
DEFINE_ANOTHER_FWK_MODULE(WriteHcalGains);
DEFINE_ANOTHER_FWK_MODULE(WriteHcalPedestals);
DEFINE_ANOTHER_FWK_MODULE(WriteHcalPedestalWidths);
DEFINE_ANOTHER_FWK_MODULE(WriteHcalQIEData);
DEFINE_ANOTHER_FWK_MODULE(WriteHcalElectronicsMap);
