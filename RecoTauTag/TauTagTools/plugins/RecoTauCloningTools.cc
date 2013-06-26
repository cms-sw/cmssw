
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "FWCore/Framework/interface/MakerMacros.h"

// Make PFJet refs from a View<Candidate> (note that the Candidates must point
// back to original PFJet refs!
//
#include "RecoTauTag/TauTagTools/interface/CastedRefProducer.h"
typedef reco::tautools::CastedRefProducer<reco::PFJetCollection,
        reco::Candidate> PFJetRefsCastFromCandView;
DEFINE_FWK_MODULE(PFJetRefsCastFromCandView);

typedef reco::tautools::CastedRefProducer<reco::PFTauCollection,
        reco::Candidate> PFTauRefsCastFromCandView;
DEFINE_FWK_MODULE(PFTauRefsCastFromCandView);

// Copy a collection of PFJet or Tau refs to a new concrete collection
#include "RecoTauTag/TauTagTools/interface/CopyProducer.h"
typedef reco::tautools::CopyProducer<reco::PFJetCollection> PFJetCopyProducer;
DEFINE_FWK_MODULE(PFJetCopyProducer);
typedef reco::tautools::CopyProducer<reco::PFTauCollection> PFTauCopyProducer;
DEFINE_FWK_MODULE(PFTauCopyProducer);
