#include "FWCore/Framework/interface/MakerMacros.h"
#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer.h"
#include "FastSimulation/Tracking/plugins/TrackCandidateProducer.h"
#include "FastSimulation/Tracking/plugins/PixelTracksProducer.h"
#include "FastSimulation/Tracking/plugins/FastTrackMerger.h"
// reco::Track accumulator:
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "FastSimulation/Tracking/plugins/RecoTrackAccumulator.h"


DEFINE_FWK_MODULE(TrajectorySeedProducer);
DEFINE_FWK_MODULE(TrackCandidateProducer);
DEFINE_FWK_MODULE(PixelTracksProducer);
DEFINE_FWK_MODULE(FastTrackMerger);
DEFINE_DIGI_ACCUMULATOR(RecoTrackAccumulator);
