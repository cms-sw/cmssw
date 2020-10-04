#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTag/Combined/interface/hiRun2LegacyCSVv2Tagger.h"

typedef JetTagComputerESProducer<hiRun2LegacyCSVv2Tagger> hiRun2LegacyCSVv2ESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(hiRun2LegacyCSVv2ESProducer);
