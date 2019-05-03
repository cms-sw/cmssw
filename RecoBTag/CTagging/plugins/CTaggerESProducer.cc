#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "RecoBTag/CTagging/interface/CharmTagger.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"

typedef JetTagComputerESProducer<CharmTagger> CharmTaggerESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CharmTaggerESProducer);
