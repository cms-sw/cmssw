#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTag/Combined/interface/HeavyIonCSVTagger.h"

typedef JetTagComputerESProducer<HeavyIonCSVTagger> HeavyIonCSVESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(HeavyIonCSVESProducer);
