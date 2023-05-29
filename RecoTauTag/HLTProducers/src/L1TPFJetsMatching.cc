#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "RecoTauTag/HLTProducers/interface/L1TJetsMatching.h"

typedef L1TJetsMatching<reco::PFJet> L1TPFJetsMatching;
DEFINE_FWK_MODULE(L1TPFJetsMatching);
