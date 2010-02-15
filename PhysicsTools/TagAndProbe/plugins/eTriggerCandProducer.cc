#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "PhysicsTools/TagAndProbe/interface/TriggerCandProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//typedef TriggerCandProducer<reco::GsfElectronCollection> eTriggerGsfElectronCollection;
typedef TriggerCandProducer<edm::View<reco::GsfElectron> > eTriggerGsfElectronCollection;
DEFINE_FWK_MODULE( eTriggerGsfElectronCollection );
typedef TriggerCandProducer<reco::CandidateCollection> eTriggerCandidateCollection;
DEFINE_ANOTHER_FWK_MODULE( eTriggerCandidateCollection );
typedef TriggerCandProducer<reco::RecoEcalCandidateCollection> eTriggerEcalCandidateCollection;
DEFINE_ANOTHER_FWK_MODULE( eTriggerEcalCandidateCollection );
typedef TriggerCandProducer<reco::RecoChargedCandidateCollection> eTriggerChargedCandidateCollection;
DEFINE_ANOTHER_FWK_MODULE( eTriggerChargedCandidateCollection );
typedef TriggerCandProducer<reco::IsolatedPixelTrackCandidateCollection> eTriggerIsolatedPixelTrackCandidateCollection;
DEFINE_ANOTHER_FWK_MODULE( eTriggerIsolatedPixelTrackCandidateCollection );
typedef TriggerCandProducer<reco::ElectronCollection> eTriggerElectronCollection;
DEFINE_ANOTHER_FWK_MODULE( eTriggerElectronCollection );

