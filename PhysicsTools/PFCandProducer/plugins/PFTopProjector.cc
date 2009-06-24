#include "PhysicsTools/PFCandProducer/interface/PFTopProjector.h"


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

typedef PFTopProjector<PFJet, PFCandidate> PFTopProjectorPFJetsOnPFCandidates;
typedef PFTopProjector<PFCandidate, PFCandidate> PFTopProjectorPFCandidatesOnPFCandidates;
typedef PFTopProjector<PileUpPFCandidate, PFCandidate> PFTopProjectorPileUpPFCandidatesOnPFCandidates;
typedef PFTopProjector<IsolatedPFCandidate, PFCandidate> PFTopProjectorIsolatedPFCandidatesOnPFCandidates;
typedef PFTopProjector<PFJet, PFCandidate> PFTopProjectorPFJetsOnPFCandidates;
typedef PFTopProjector<PFTau, PFJet> PFTopProjectorPFTausOnPFJets;

