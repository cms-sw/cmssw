#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoBTag/SoftLepton/interface/SoftLepton.h"
#include "RecoBTag/SoftLepton/interface/SoftElectronProducer.h"

#include "RecoBTag/SoftLepton/interface/ElectronTagger.h"
#include "RecoBTag/SoftLepton/interface/MuonTagger.h"
#include "RecoBTag/SoftLepton/interface/MuonTaggerNoIP.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerDistance.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerByPt.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerESProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SoftLepton);
DEFINE_ANOTHER_FWK_MODULE(SoftElectronProducer);

typedef LeptonTaggerESProducer<ElectronTagger>       ElectronTaggerESProducer;
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(ElectronTaggerESProducer);

typedef LeptonTaggerESProducer<MuonTagger>           MuonTaggerESProducer;
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(MuonTaggerESProducer);

typedef LeptonTaggerESProducer<MuonTaggerNoIP>       MuonTaggerNoIPESProducer;
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(MuonTaggerNoIPESProducer);

typedef LeptonTaggerESProducer<LeptonTaggerDistance> LeptonTaggerDistanceESProducer;
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(LeptonTaggerDistanceESProducer);

typedef LeptonTaggerESProducer<LeptonTaggerByPt>     LeptonTaggerByPtESProducer;
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(LeptonTaggerByPtESProducer);
