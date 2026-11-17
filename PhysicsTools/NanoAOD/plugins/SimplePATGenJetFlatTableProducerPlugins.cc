// NanoAOD table producer for pat::GenJet, counterpart of
// PhysicsTools/NanoAOD/plugins/SimplePATFlatTableProducerPlugins.cc:16
//   typedef SimpleFlatTableProducer<pat::Jet> SimplePATJetFlatTableProducer;
#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"
#include "DataFormats/PatCandidates/interface/GenJet.h"

typedef SimpleFlatTableProducer<pat::GenJet> SimplePATGenJetFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimplePATGenJetFlatTableProducer);
