#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/ESRawToDigi/interface/ESRawToDigi.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ESRawToDigi);

#include "EventFilter/ESRawToDigi/interface/ESUnpackerWorkerESProducer.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(ESUnpackerWorkerESProducer);


