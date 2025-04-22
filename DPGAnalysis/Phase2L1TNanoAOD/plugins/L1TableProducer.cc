#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/L1Trigger/interface/VertexWord.h"
typedef SimpleFlatTableProducer<l1t::VertexWord> SimpleL1VtxWordCandidateFlatTableProducer;

#include "DataFormats/L1Trigger/interface/P2GTAlgoBlock.h"
typedef SimpleFlatTableProducer<l1t::P2GTAlgoBlock> P2GTAlgoBlockFlatTableProducer;

#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
typedef SimpleFlatTableProducer<l1t::P2GTCandidate> SimpleP2GTCandidateFlatTableProducer;

#include "DataFormats/L1Trigger/interface/TkJetWord.h"
typedef SimpleFlatTableProducer<l1t::TkJetWord> SimpleL1TkJetWordCandidateFlatTableProducer;

#include "DataFormats/L1Trigger/interface/TkTripletWord.h"
typedef SimpleFlatTableProducer<l1t::TkTripletWord> SimpleTkTripletWordCandidateFlatTableProducer;

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
typedef SimpleFlatTableProducer<l1t::L1Candidate> SimpleTriggerL1CandidateFlatTableProducer;

#include "DataFormats/L1TCorrelator/interface/TkEm.h"
typedef SimpleFlatTableProducer<l1t::TkEm> SimpleTriggerL1TkEmFlatTableProducer;

#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
typedef SimpleFlatTableProducer<l1t::TkElectron> SimpleTriggerL1TkElectronFlatTableProducer;

#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"
typedef SimpleFlatTableProducer<l1t::SAMuon> SimpleTriggerL1SAMuonFlatTableProducer;

#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
typedef SimpleFlatTableProducer<l1t::TrackerMuon> SimpleTriggerL1TrackerMuonFlatTableProducer;

#include "DataFormats/L1TCalorimeterPhase2/interface/Phase2L1CaloJet.h"
typedef SimpleFlatTableProducer<l1tp2::Phase2L1CaloJet> SimpleTriggerL1CaloJetFlatTableProducer;

#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
typedef SimpleFlatTableProducer<l1t::PFTau> SimpleTriggerL1PFTauFlatTableProducer;

#include "DataFormats/L1TParticleFlow/interface/HPSPFTau.h"
typedef SimpleFlatTableProducer<l1t::HPSPFTau> SimpleTriggerL1HPSPFTauFlatTableProducer;

#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
typedef SimpleFlatTableProducer<l1t::PFJet> SimpleTriggerL1PFJetFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleL1VtxWordCandidateFlatTableProducer);
DEFINE_FWK_MODULE(P2GTAlgoBlockFlatTableProducer);
DEFINE_FWK_MODULE(SimpleP2GTCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleL1TkJetWordCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTkTripletWordCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1CandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1TkEmFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1TkElectronFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1SAMuonFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1TrackerMuonFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1CaloJetFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1PFTauFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1HPSPFTauFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1PFJetFlatTableProducer);
