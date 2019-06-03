#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"



#include "CalibTracker/SiStripESProducers/plugins/real/SiStripQualityESProducer.h"
DEFINE_FWK_EVENTSETUP_MODULE(SiStripQualityESProducer);

#include "CalibTracker/SiStripESProducers/plugins/real/SiStripLorentzAngleDepESProducer.h"
DEFINE_FWK_EVENTSETUP_MODULE(SiStripLorentzAngleDepESProducer);
