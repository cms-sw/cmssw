#include "FWCore/Framework/interface/MakerMacros.h"

// This is the analyzer
#include "L1TriggerOffline/L1Analyzer/interface/L1Analyzer.h"

// These for L1 collections
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

// This for RECO tau collection
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

// This to convert to candidates
#include "PhysicsTools/CandAlgos/interface/ShallowCloneProducer.h"
#include "L1TriggerOffline/L1Analyzer/interface/TagCaloJetProducer.h"

// L1 converters
typedef ShallowCloneProducer<l1extra::L1EmParticleCollection> L1EmParticleShallowCloneProducer;
typedef ShallowCloneProducer<l1extra::L1JetParticleCollection> L1JetParticleShallowCloneProducer;
typedef ShallowCloneProducer<l1extra::L1EtMissParticleCollection> L1EtMissParticleShallowCloneProducer;
typedef ShallowCloneProducer<l1extra::L1MuonParticleCollection> L1MuonParticleShallowCloneProducer;

// Reco converter
typedef TagCaloJetProducer TauCaloJetProducer;

DEFINE_FWK_MODULE(L1Analyzer);
DEFINE_FWK_MODULE(TauCaloJetProducer);
DEFINE_FWK_MODULE(L1EmParticleShallowCloneProducer);
DEFINE_FWK_MODULE(L1JetParticleShallowCloneProducer);
DEFINE_FWK_MODULE(L1MuonParticleShallowCloneProducer);
DEFINE_FWK_MODULE(L1EtMissParticleShallowCloneProducer);



