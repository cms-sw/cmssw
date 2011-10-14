#include "PhysicsTools/PatUtils/interface/ShiftedJetProducerT.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

typedef ShiftedJetProducerT<pat::Jet> ShiftedPATJetProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedPATJetProducer);
