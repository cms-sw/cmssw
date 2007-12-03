#include "FWCore/Framework/interface/MakerMacros.h"
#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer.h"
#include "FastSimulation/Tracking/plugins/TrackCandidateProducer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(TrajectorySeedProducer);
DEFINE_ANOTHER_FWK_MODULE(TrackCandidateProducer);
