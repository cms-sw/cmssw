#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"

#include "RecoBTag/Combined/interface/CandidateChargeBTagComputer.h"

typedef JetTagComputerESProducer<CandidateChargeBTagComputer> CandidateChargeBTagESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CandidateChargeBTagESProducer);
