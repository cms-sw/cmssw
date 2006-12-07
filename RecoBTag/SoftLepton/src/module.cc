#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoBTag/SoftLepton/interface/SoftLepton.h"
#include "RecoBTag/SoftLepton/interface/MuonTaggerESProducer.h"
#include "RecoBTag/SoftLepton/interface/MuonTaggerNoIPESProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SoftLepton);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(MuonTaggerESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(MuonTaggerNoIPESProducer);
