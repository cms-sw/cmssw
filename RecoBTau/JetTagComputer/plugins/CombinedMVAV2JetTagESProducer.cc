#include "FWCore/Framework/interface/ModuleFactory.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTag/Combined/interface/CombinedMVAV2JetTagComputer.h"

typedef JetTagComputerESProducer<CombinedMVAV2JetTagComputer> CombinedMVAV2JetTagESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CombinedMVAV2JetTagESProducer);
