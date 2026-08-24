#ifndef RecoTracker_FinalTrackSelectors_TrackTorchClassifierFeaturesSoA_h
#define RecoTracker_FinalTrackSelectors_TrackTorchClassifierFeaturesSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(TrackTorchClassifierFeaturesSoALayout,
                    SOA_COLUMN(float, dxyBeamSpot),
                    SOA_COLUMN(float, dzBeamSpot),
                    SOA_COLUMN(float, dxyError),
                    SOA_COLUMN(float, dzError),
                    SOA_COLUMN(float, normalizedChi2),
                    SOA_COLUMN(float, eta),
                    SOA_COLUMN(float, phi),
                    SOA_COLUMN(float, etaError),
                    SOA_COLUMN(float, phiError),
                    SOA_COLUMN(float, ndof),
                    SOA_COLUMN(float, lostInnerHits),
                    SOA_COLUMN(float, lostOuterHits),
                    SOA_COLUMN(float, layersWithoutMeas),
                    SOA_COLUMN(float, validPixelHits),
                    SOA_COLUMN(float, validStripHits))

using TrackTorchClassifierFeaturesSoA = TrackTorchClassifierFeaturesSoALayout<>;

// Define the SoA layout for track scores (output)
GENERATE_SOA_LAYOUT(TrackTorchClassifierScoresSoALayout, SOA_COLUMN(float, score))

using TrackTorchClassifierScoresSoA = TrackTorchClassifierScoresSoALayout<>;

#endif  // RecoTracker_FinalTrackSelectors_TrackTorchClassifierFeaturesSoA_h
