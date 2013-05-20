#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SoftLepton.h"
#include "SoftPFElectronTagInfoProducer.h"
#include "SoftPFMuonTagInfoProducer.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTag/SoftLepton/interface/ElectronTagger.h"
#include "RecoBTag/SoftLepton/interface/MuonTagger.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerByPt.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerByIP.h"

DEFINE_FWK_MODULE(SoftPFElectronTagInfoProducer);
DEFINE_FWK_MODULE(SoftPFMuonTagInfoProducer);

typedef JetTagComputerESProducer<ElectronTagger>        ElectronTaggerESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(ElectronTaggerESProducer);

typedef JetTagComputerESProducer<MuonTagger>            MuonTaggerESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(MuonTaggerESProducer);

typedef JetTagComputerESProducer<LeptonTaggerByPt>      LeptonTaggerByPtESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(LeptonTaggerByPtESProducer);

typedef JetTagComputerESProducer<LeptonTaggerByIP>      LeptonTaggerByIPESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(LeptonTaggerByIPESProducer);
