#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "RecoTauTag/HLTProducers/interface/L1TJetsMatching.h"

using L1TCaloJetsMatching = L1TJetsMatching<reco::CaloJet>;
DEFINE_FWK_MODULE(L1TCaloJetsMatching);
