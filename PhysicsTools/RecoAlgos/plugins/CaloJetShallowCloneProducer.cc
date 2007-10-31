#include "DataFormats/JetReco/interface/CaloJet.h"
#include "PhysicsTools/CandAlgos/interface/ShallowCloneProducer.h"

typedef ShallowCloneProducer<reco::CaloJetCollection> CaloJetShallowCloneProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( CaloJetShallowCloneProducer );
