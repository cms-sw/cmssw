#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMET/METProducers/interface/METProducer.h"
#include "RecoMET/METProducers/interface/BeamHaloSummaryProducer.h"
#include "RecoMET/METProducers/interface/CSCHaloDataProducer.h" 
#include "RecoMET/METProducers/interface/HcalHaloDataProducer.h" 
#include "RecoMET/METProducers/interface/EcalHaloDataProducer.h" 
#include "RecoMET/METProducers/interface/GlobalHaloDataProducer.h" 
#include "RecoMET/METProducers/interface/ParticleFlowForChargedMETProducer.h" 

using cms::METProducer;
using reco::BeamHaloSummaryProducer;
using reco::CSCHaloDataProducer;
using reco::HcalHaloDataProducer;
using reco::EcalHaloDataProducer;
using reco::GlobalHaloDataProducer;
using reco::ParticleFlowForChargedMETProducer;

DEFINE_FWK_MODULE(METProducer);
DEFINE_FWK_MODULE(BeamHaloSummaryProducer);
DEFINE_FWK_MODULE(CSCHaloDataProducer);
DEFINE_FWK_MODULE(HcalHaloDataProducer);
DEFINE_FWK_MODULE(EcalHaloDataProducer);
DEFINE_FWK_MODULE(GlobalHaloDataProducer);
DEFINE_FWK_MODULE(ParticleFlowForChargedMETProducer);

#include "RecoMET/METProducers/interface/MuonMET.h"
using cms::MuonMET;
DEFINE_FWK_MODULE(MuonMET);
#include "RecoMET/METProducers/interface/MuonMETValueMapProducer.h"
using cms::MuonMETValueMapProducer;
DEFINE_FWK_MODULE(MuonMETValueMapProducer);
#include "RecoMET/METProducers/interface/MuonTCMETValueMapProducer.h"
using cms::MuonTCMETValueMapProducer;
DEFINE_FWK_MODULE(MuonTCMETValueMapProducer);
