#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAJetTagComputer.h"

DEFINE_SEAL_MODULE();

typedef JetTagComputerESProducer<GenericMVAJetTagComputer>       GenericMVAJetTagESProducer;
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(GenericMVAJetTagESProducer);

