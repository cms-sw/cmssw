#include "FWCore/Framework/interface/MakerMacros.h"
#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer.h"
#include "FastSimulation/Tracking/plugins/TrackCandidateProducer.h"
#include "FastSimulation/Tracking/plugins/PixelTracksProducer.h"
#include "FastSimulation/Tracking/plugins/FastTrackMerger.h"



DEFINE_FWK_MODULE(TrajectorySeedProducer);
DEFINE_FWK_MODULE(TrackCandidateProducer);
DEFINE_FWK_MODULE(PixelTracksProducer);
DEFINE_FWK_MODULE(FastTrackMerger);
