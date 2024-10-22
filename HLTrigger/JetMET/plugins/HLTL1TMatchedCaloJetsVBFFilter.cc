#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTL1TMatchedJetsVBFFilter.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

typedef HLTL1TMatchedJetsVBFFilter<reco::CaloJet> HLTL1TMatchedCaloJetsVBFFilter;
DEFINE_FWK_MODULE(HLTL1TMatchedCaloJetsVBFFilter);
