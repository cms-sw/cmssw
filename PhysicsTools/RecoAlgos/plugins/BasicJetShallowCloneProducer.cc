#include "DataFormats/JetReco/interface/BasicJet.h"
#include "PhysicsTools/CandAlgos/interface/ShallowCloneProducer.h"

typedef ShallowCloneProducer<reco::BasicJetCollection> BasicJetShallowCloneProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( BasicJetShallowCloneProducer );
