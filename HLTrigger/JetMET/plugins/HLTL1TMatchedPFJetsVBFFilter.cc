#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTL1TMatchedJetsVBFFilter.h"
#include "DataFormats/JetReco/interface/PFJet.h"

typedef HLTL1TMatchedJetsVBFFilter<reco::PFJet> HLTL1TMatchedPFJetsVBFFilter;
DEFINE_FWK_MODULE(HLTL1TMatchedPFJetsVBFFilter);
