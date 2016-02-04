#include "DPGAnalysis/Skims/interface/TriggerMatchProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/MuonReco/interface/Muon.h"
typedef TriggerMatchProducer< reco::Muon > trgMatchMuonProducer;
DEFINE_FWK_MODULE( trgMatchMuonProducer );

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
typedef TriggerMatchProducer< reco::GsfElectron > trgMatchGsfElectronProducer;
DEFINE_FWK_MODULE( trgMatchGsfElectronProducer );

#include "DataFormats/JetReco/interface/Jet.h"
typedef TriggerMatchProducer< reco::Jet > trgMatchJetProducer;
DEFINE_FWK_MODULE( trgMatchJetProducer );

#include "DataFormats/METReco/interface/MET.h"
typedef TriggerMatchProducer< reco::MET > trgMatchMETProducer;
DEFINE_FWK_MODULE( trgMatchMETProducer );

#include "DataFormats/Candidate/interface/Candidate.h"
typedef TriggerMatchProducer<reco::Candidate> trgMatchCandidateProducer;
DEFINE_FWK_MODULE( trgMatchCandidateProducer );

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
typedef TriggerMatchProducer<reco::RecoEcalCandidate> trgMatchEcalCandidateProducer;
DEFINE_FWK_MODULE( trgMatchEcalCandidateProducer );

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
typedef TriggerMatchProducer<reco::RecoChargedCandidate> trgMatchChargedCandidateProducer;
DEFINE_FWK_MODULE( trgMatchChargedCandidateProducer );

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
typedef TriggerMatchProducer<reco::IsolatedPixelTrackCandidate> trgMatchIsolatedPixelTrackCandidateProducer;
DEFINE_FWK_MODULE( trgMatchIsolatedPixelTrackCandidateProducer );

#include "DataFormats/EgammaCandidates/interface/Electron.h"
typedef TriggerMatchProducer<reco::Electron> trgMatchElectronProducer;
DEFINE_FWK_MODULE( trgMatchElectronProducer );

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
typedef TriggerMatchProducer< reco::SuperCluster > trgMatchSuperClusterProducer;
DEFINE_FWK_MODULE( trgMatchSuperClusterProducer );
