#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTag/Combined/interface/heavyIonCSVTagger.h"

typedef JetTagComputerESProducer<HeavyIonCSVTagger> heavyIonCSVESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(heavyIonCSVESProducer);
