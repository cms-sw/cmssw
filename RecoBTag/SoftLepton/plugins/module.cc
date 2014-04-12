#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SoftLepton.h"
#include "SoftPFElectronTagInfoProducer.h"
#include "SoftPFMuonTagInfoProducer.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTag/SoftLepton/interface/ElectronTagger.h"
#include "RecoBTag/SoftLepton/interface/MuonTagger.h"
#include "RecoBTag/SoftLepton/interface/MuonTaggerNoIP.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerDistance.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerByPt.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerByIP.h"


DEFINE_FWK_MODULE(SoftLepton);
DEFINE_FWK_MODULE(SoftPFElectronTagInfoProducer);
DEFINE_FWK_MODULE(SoftPFMuonTagInfoProducer);

typedef JetTagComputerESProducer<ElectronTagger>        ElectronTaggerESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(ElectronTaggerESProducer);

typedef JetTagComputerESProducer<MuonTagger>            MuonTaggerESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(MuonTaggerESProducer);

typedef JetTagComputerESProducer<LeptonTaggerDistance>  LeptonTaggerByDistanceESProducer;  //DO NOT REMOVE, CALLED BY TRIGGERS
DEFINE_FWK_EVENTSETUP_MODULE(LeptonTaggerByDistanceESProducer);

typedef JetTagComputerESProducer<MuonTaggerNoIP>        MuonTaggerNoIPESProducer;  // DO NOT REMOVE, CALLED BY Triggers
DEFINE_FWK_EVENTSETUP_MODULE(MuonTaggerNoIPESProducer);

typedef JetTagComputerESProducer<LeptonTaggerByPt>      LeptonTaggerByPtESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(LeptonTaggerByPtESProducer);

typedef JetTagComputerESProducer<LeptonTaggerByIP>      LeptonTaggerByIPESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(LeptonTaggerByIPESProducer);
