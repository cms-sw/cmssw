#include "FWCore/Framework/interface/MakerMacros.h"

// L1 extra shallow clone producers
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

#include "CommonTools/CandAlgos/interface/ShallowCloneProducer.h"

typedef ShallowCloneProducer<l1extra::L1EmParticleCollection> L1EmParticleShallowCloneProducer;
typedef ShallowCloneProducer<l1extra::L1JetParticleCollection> L1JetParticleShallowCloneProducer;
typedef ShallowCloneProducer<l1extra::L1EtMissParticleCollection> L1EtMissParticleShallowCloneProducer;
typedef ShallowCloneProducer<l1extra::L1MuonParticleCollection> L1MuonParticleShallowCloneProducer;

DEFINE_FWK_MODULE(L1EmParticleShallowCloneProducer);
DEFINE_FWK_MODULE(L1JetParticleShallowCloneProducer);
DEFINE_FWK_MODULE(L1MuonParticleShallowCloneProducer);
DEFINE_FWK_MODULE(L1EtMissParticleShallowCloneProducer);

// tau tagged jet producer
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "L1TriggerOffline/L1Analyzer/interface/TagCaloJetProducer.h"

typedef TagCaloJetProducer TauCaloJetProducer;

DEFINE_FWK_MODULE(TauCaloJetProducer);

// special selector
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/View.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/PtMinSelector.h"
#include "CommonTools/UtilAlgos/interface/EtaRangeSelector.h"
#include "CommonTools/UtilAlgos/interface/PdgIdSelector.h"
#include "CommonTools/UtilAlgos/interface/AndSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector<
          edm::View<reco::Candidate>,
          AndSelector<
            PtMinSelector,
            EtaRangeSelector,
            PdgIdSelector
	    >,
          reco::CandidateCollection
        > EtaPtMinPdgIdCandViewSelector;

DEFINE_FWK_MODULE( EtaPtMinPdgIdCandViewSelector );


