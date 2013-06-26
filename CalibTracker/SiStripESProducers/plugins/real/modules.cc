#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"



#include "CalibTracker/SiStripESProducers/plugins/real/SiStripQualityESProducer.h"
DEFINE_FWK_EVENTSETUP_MODULE(SiStripQualityESProducer);

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CalibTracker/SiStripESProducers/plugins/real/SiStripGainESProducerTemplate.h"
typedef SiStripGainESProducerTemplate<SiStripGainRcd,SiStripApvGainRcd> SiStripGainESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(SiStripGainESProducer);
typedef SiStripGainESProducerTemplate<SiStripGainSimRcd,SiStripApvGainSimRcd> SiStripGainSimESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(SiStripGainSimESProducer);

#include "CalibTracker/SiStripESProducers/plugins/real/SiStripDelayESProducer.h"
DEFINE_FWK_EVENTSETUP_MODULE(SiStripDelayESProducer);

#include "CalibTracker/SiStripESProducers/plugins/real/SiStripLorentzAngleDepESProducer.h"
DEFINE_FWK_EVENTSETUP_MODULE(SiStripLorentzAngleDepESProducer);

#include "CalibTracker/SiStripESProducers/plugins/real/SiStripBackPlaneCorrectionDepESProducer.h"
DEFINE_FWK_EVENTSETUP_MODULE(SiStripBackPlaneCorrectionDepESProducer);
