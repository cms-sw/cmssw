#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"
typedef SimpleFlatTableProducer<pat::CompositeCandidate>
    SimpleCompositeCandidateFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleCompositeCandidateFlatTableProducer);
