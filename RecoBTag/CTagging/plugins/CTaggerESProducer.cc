#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTag/CTagging/interface/CharmTagger.h"

typedef JetTagComputerESProducer<CharmTagger>        CharmTaggerESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CharmTaggerESProducer);
