#include "DataFormats/JetReco/interface/GenJet.h"
#include "PhysicsTools/CandAlgos/interface/ShallowCloneProducer.h"

typedef ShallowCloneProducer<reco::GenJetCollection> GenJetShallowCloneProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenJetShallowCloneProducer );
