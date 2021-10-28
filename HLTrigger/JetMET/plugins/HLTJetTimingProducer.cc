#include "HLTJetTimingProducer.h"

typedef HLTJetTimingProducer<reco::CaloJet> HLTCaloJetTimingProducer;
typedef HLTJetTimingProducer<reco::PFJet> HLTPFJetTimingProducer;

// declare classes as framework plugins
DEFINE_FWK_MODULE(HLTCaloJetTimingProducer);
DEFINE_FWK_MODULE(HLTPFJetTimingProducer);
