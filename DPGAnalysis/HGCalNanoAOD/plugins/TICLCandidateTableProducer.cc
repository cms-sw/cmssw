#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
typedef SimpleCollectionFlatTableProducer<TICLCandidate> TICLCandidateTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TICLCandidateTableProducer);
