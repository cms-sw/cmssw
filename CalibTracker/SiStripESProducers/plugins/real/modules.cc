#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"

DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripESProducers/plugins/real/SiStripQualityESProducer.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(SiStripQualityESProducer);

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CalibTracker/SiStripESProducers/plugins/real/SiStripGainESProducerTemplate.h"
typedef SiStripGainESProducerTemplate<SiStripGainRcd,SiStripApvGainRcd> SiStripGainESProducer;
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(SiStripGainESProducer);
typedef SiStripGainESProducerTemplate<SiStripGainSimRcd,SiStripApvGainSimRcd> SiStripGainSimESProducer;
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(SiStripGainSimESProducer);

