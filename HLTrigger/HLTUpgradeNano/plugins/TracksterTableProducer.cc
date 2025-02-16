#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
typedef SimpleFlatTableProducer<ticl::Trackster> TracksterTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TracksterTableProducer);
