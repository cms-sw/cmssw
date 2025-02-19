#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/ESRawToDigi/interface/ESRawToDigi.h"


DEFINE_FWK_MODULE(ESRawToDigi);

#include "EventFilter/ESRawToDigi/interface/ESUnpackerWorkerESProducer.h"
DEFINE_FWK_EVENTSETUP_MODULE(ESUnpackerWorkerESProducer);


