#include "CommonTools/ParticleFlow/plugins/TopProjector.h"


#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

//TODO just for testing, remove this
#include "DataFormats/TrackReco/interface/Track.h"

#include "FWCore/Framework/interface/ESHandle.h"

// #include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Math/interface/deltaR.h"


using namespace std;
using namespace edm;
using namespace reco;

typedef TopProjector<PFJet, PFCandidate> TPPFJetsOnPFCandidates;
typedef TopProjector<PFCandidate, PFCandidate> TPPFCandidatesOnPFCandidates;
typedef TopProjector<PileUpPFCandidate, PFCandidate> TPPileUpPFCandidatesOnPFCandidates;
typedef TopProjector<PFCandidate, PileUpPFCandidate> TPPFCandidatesOnPileUpPFCandidates;
typedef TopProjector<IsolatedPFCandidate, PFCandidate> TPIsolatedPFCandidatesOnPFCandidates;
typedef TopProjector<PFJet, PFCandidate> TPPFJetsOnPFCandidates;
typedef TopProjector<PFTau, PFJet> TPPFTausOnPFJets;

