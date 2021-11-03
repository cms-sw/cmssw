#include "HLTJetTimingFilter.h"

typedef HLTJetTimingFilter<reco::CaloJet> HLTCaloJetTimingFilter;
typedef HLTJetTimingFilter<reco::PFJet> HLTPFJetTimingFilter;

// declare classes as framework plugins
DEFINE_FWK_MODULE(HLTCaloJetTimingFilter);
DEFINE_FWK_MODULE(HLTPFJetTimingFilter);
