#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMET/METProducers/interface/METProducer.h"
#include "RecoMET/METProducers/interface/BeamHaloSummaryProducer.h"
#include "RecoMET/METProducers/interface/CSCHaloDataProducer.h" 
#include "RecoMET/METProducers/interface/HcalHaloDataProducer.h" 
#include "RecoMET/METProducers/interface/EcalHaloDataProducer.h" 
#include "RecoMET/METProducers/interface/GlobalHaloDataProducer.h" 
#include "RecoMET/METProducers/interface/PFCandidatesForTrackMETProducer.h" 
#include "RecoMET/METProducers/interface/PFMETProducerMVA.h" 
#include "RecoMET/METProducers/interface/PFMETProducerMVA2.h" 
#include "RecoMET/METProducers/interface/PFMETProducerMVAData.h" 

using cms::METProducer;
using reco::BeamHaloSummaryProducer;
using reco::CSCHaloDataProducer;
using reco::HcalHaloDataProducer;
using reco::EcalHaloDataProducer;
using reco::GlobalHaloDataProducer;
using reco::PFCandidatesForTrackMETProducer;
using reco::PFMETProducerMVA;
using reco::PFMETProducerMVA2;
using reco::PFMETProducerMVAData;

DEFINE_FWK_MODULE(METProducer);
DEFINE_FWK_MODULE(BeamHaloSummaryProducer);
DEFINE_FWK_MODULE(CSCHaloDataProducer);
DEFINE_FWK_MODULE(HcalHaloDataProducer);
DEFINE_FWK_MODULE(EcalHaloDataProducer);
DEFINE_FWK_MODULE(GlobalHaloDataProducer);
DEFINE_FWK_MODULE(PFCandidatesForTrackMETProducer);
DEFINE_FWK_MODULE(PFMETProducerMVA);
DEFINE_FWK_MODULE(PFMETProducerMVA2);
DEFINE_FWK_MODULE(PFMETProducerMVAData);


#include "RecoMET/METProducers/interface/MuonMET.h"
using cms::MuonMET;
DEFINE_FWK_MODULE(MuonMET);
#include "RecoMET/METProducers/interface/MuonMETValueMapProducer.h"
using cms::MuonMETValueMapProducer;
DEFINE_FWK_MODULE(MuonMETValueMapProducer);
#include "RecoMET/METProducers/interface/MuonTCMETValueMapProducer.h"
using cms::MuonTCMETValueMapProducer;
DEFINE_FWK_MODULE(MuonTCMETValueMapProducer);
