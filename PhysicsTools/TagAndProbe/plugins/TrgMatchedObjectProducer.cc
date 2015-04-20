




#include "PhysicsTools/TagAndProbe/interface/TriggerCandProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/MuonReco/interface/Muon.h"
typedef TriggerCandProducer< reco::Muon > trgMatchedMuonProducer;
DEFINE_FWK_MODULE( trgMatchedMuonProducer );

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
typedef TriggerCandProducer< reco::GsfElectron > trgMatchedGsfElectronProducer;
DEFINE_FWK_MODULE( trgMatchedGsfElectronProducer );

#include "DataFormats/PatCandidates/interface/Electron.h"
typedef TriggerCandProducer< reco::GsfElectron > trgMatchedPatElectronProducer;
DEFINE_FWK_MODULE( trgMatchedPatElectronProducer );

#include "DataFormats/JetReco/interface/Jet.h"
typedef TriggerCandProducer< reco::Jet > trgMatchedJetProducer;
DEFINE_FWK_MODULE( trgMatchedJetProducer );

#include "DataFormats/METReco/interface/MET.h"
typedef TriggerCandProducer< reco::MET > trgMatchedMETProducer;
DEFINE_FWK_MODULE( trgMatchedMETProducer );

#include "DataFormats/Candidate/interface/Candidate.h"
typedef TriggerCandProducer<reco::Candidate> trgMatchedCandidateProducer;
DEFINE_FWK_MODULE( trgMatchedCandidateProducer );

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
typedef TriggerCandProducer<reco::RecoEcalCandidate> trgMatchedEcalCandidateProducer;
DEFINE_FWK_MODULE( trgMatchedEcalCandidateProducer );

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
typedef TriggerCandProducer<reco::RecoChargedCandidate> trgMatchedChargedCandidateProducer;
DEFINE_FWK_MODULE( trgMatchedChargedCandidateProducer );

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
typedef TriggerCandProducer<reco::IsolatedPixelTrackCandidate> trgMatchedIsolatedPixelTrackCandidateProducer;
DEFINE_FWK_MODULE( trgMatchedIsolatedPixelTrackCandidateProducer );

#include "DataFormats/EgammaCandidates/interface/Electron.h"
typedef TriggerCandProducer<reco::Electron> trgMatchedElectronProducer;
DEFINE_FWK_MODULE( trgMatchedElectronProducer );

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
typedef TriggerCandProducer< reco::SuperCluster > trgMatchedSuperClusterProducer;
DEFINE_FWK_MODULE( trgMatchedSuperClusterProducer );
#include "DataFormats/EgammaCandidates/interface/Photon.h"
typedef TriggerCandProducer< reco::Photon > trgMatchedPhotonProducer;
DEFINE_FWK_MODULE( trgMatchedPhotonProducer );
