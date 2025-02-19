#include "PhysicsTools/PatUtils/interface/ShiftedJetProducerT.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"

typedef ShiftedJetProducerT<pat::Jet, JetCorrExtractorT<pat::Jet> > ShiftedPATJetProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedPATJetProducer);
