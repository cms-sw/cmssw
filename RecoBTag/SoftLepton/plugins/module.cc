#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoBTag/SoftLepton/plugins/SoftLepton.h"
#include "RecoBTag/SoftLepton/plugins/SoftPFElectronTagInfoProducer.h"
#include "RecoBTag/SoftLepton/plugins/SoftPFMuonTagInfoProducer.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoBTag/SoftLepton/interface/GenericSelectorByValueMap.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerByIP.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerByPt.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerDistance.h"
#include "RecoBTag/SoftLepton/interface/ElectronTagger.h"
#include "RecoBTag/SoftLepton/interface/MuonTagger.h"
#include "RecoBTag/SoftLepton/interface/MuonTaggerNoIP.h"


DEFINE_FWK_MODULE(SoftLepton);
DEFINE_FWK_MODULE(SoftPFElectronTagInfoProducer);
DEFINE_FWK_MODULE(SoftPFMuonTagInfoProducer);

// "float" is the type stored in the ValueMap
typedef edm::GenericSelectorByValueMap<reco::GsfElectron, float> BtagGsfElectronSelector;
DEFINE_FWK_MODULE(BtagGsfElectronSelector);


typedef JetTagComputerESProducer<LeptonTaggerByIP>      LeptonTaggerByIPESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(LeptonTaggerByIPESProducer);

typedef JetTagComputerESProducer<LeptonTaggerByPt>      LeptonTaggerByPtESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(LeptonTaggerByPtESProducer);

typedef JetTagComputerESProducer<LeptonTaggerDistance>  LeptonTaggerByDistanceESProducer;  //DO NOT REMOVE, CALLED BY TRIGGERS
DEFINE_FWK_EVENTSETUP_MODULE(LeptonTaggerByDistanceESProducer);

typedef JetTagComputerESProducer<ElectronTagger>        ElectronTaggerESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(ElectronTaggerESProducer);

typedef JetTagComputerESProducer<MuonTagger>            MuonTaggerESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(MuonTaggerESProducer);

typedef JetTagComputerESProducer<MuonTaggerNoIP>        MuonTaggerNoIPESProducer;  // DO NOT REMOVE, CALLED BY Triggers
DEFINE_FWK_EVENTSETUP_MODULE(MuonTaggerNoIPESProducer);
