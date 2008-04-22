#include "FWCore/Framework/interface/ModuleFactory.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTau/JetTagComputer/interface/CombinedMVAJetTagComputer.h"

typedef JetTagComputerESProducer<CombinedMVAJetTagComputer> CombinedMVAJetTagESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CombinedMVAJetTagESProducer);
