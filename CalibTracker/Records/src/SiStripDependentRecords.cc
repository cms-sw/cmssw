#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

EVENTSETUP_RECORD_REG(SiStripDetCablingRcd);
EVENTSETUP_RECORD_REG(SiStripFecCablingRcd);
EVENTSETUP_RECORD_REG(SiStripRegionCablingRcd);

EVENTSETUP_RECORD_REG(SiStripGainRcd);
EVENTSETUP_RECORD_REG(SiStripGainSimRcd);

EVENTSETUP_RECORD_REG(SiStripQualityRcd);

EVENTSETUP_RECORD_REG(SiStripDelayRcd);
EVENTSETUP_RECORD_REG(SiStripLorentzAngleDepRcd);
EVENTSETUP_RECORD_REG(SiStripBackPlaneCorrectionDepRcd);
EVENTSETUP_RECORD_REG(SiStripHashedDetIdRcd);
EVENTSETUP_RECORD_REG(SiStripNoisesDepRcd);
EVENTSETUP_RECORD_REG(SiStripBadModuleDepRcd);
