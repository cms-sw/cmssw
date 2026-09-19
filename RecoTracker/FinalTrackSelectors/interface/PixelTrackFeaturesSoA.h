#ifndef RecoTracker_FinalTrackSelectors_PixelTrackFeaturesSoA_h
#define RecoTracker_FinalTrackSelectors_PixelTrackFeaturesSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

// The column order below is the positional input ABI of the models: every model consumes a prefix
// of it, so appending a column never changes an existing model's score. The columns from caFitChi2
// on are written only when useHitFeatures is true.
GENERATE_SOA_LAYOUT(PixelTrackFeaturesSoALayout,
                    SOA_COLUMN(float, chi2),
                    SOA_COLUMN(float, dzError),
                    SOA_COLUMN(float, dxyError),
                    SOA_COLUMN(float, eta),
                    SOA_COLUMN(float, nHits),
                    SOA_COLUMN(float, phi),
                    SOA_COLUMN(float, phiError),
                    SOA_COLUMN(float, pt),
                    SOA_COLUMN(float, qOverPtError),
                    SOA_COLUMN(float, dzBS),
                    SOA_COLUMN(float, dxyBS),
                    SOA_COLUMN(float, nLayers),
                    SOA_COLUMN(float, cotThetaError),
                    SOA_COLUMN(float, covCotThetaDz),
                    SOA_COLUMN(float, covDxyQOverPt),
                    SOA_COLUMN(float, covPhiDxy),
                    SOA_COLUMN(float, covPhiQOverPt),
                    SOA_COLUMN(float, caFitChi2),
                    SOA_COLUMN(float, psFrac),
                    SOA_COLUMN(float, r0),
                    SOA_COLUMN(float, nPS),
                    SOA_COLUMN(float, spanZ),
                    SOA_COLUMN(float, nStubs),
                    SOA_COLUMN(float, logChi2Stub),
                    SOA_COLUMN(float, kErr),
                    SOA_COLUMN(float, dcaEst),
                    SOA_COLUMN(float, nBarrel),
                    SOA_COLUMN(float, rzChi2),
                    SOA_COLUMN(float, meanStubKappa),
                    // leverArm = rMax - r0, with rMax the largest hit radius.
                    SOA_COLUMN(float, leverArm),
                    SOA_COLUMN(float, rMax),
                    // Merged-collection provenance: nAttached and nOTExtras count the per-track CSR
                    // hit span; iterationId is constant on a single-arm collection.
                    SOA_COLUMN(float, nAttached),
                    SOA_COLUMN(float, nOTExtras),
                    SOA_COLUMN(float, iterationId),
                    SOA_COLUMN(float, ndof),
                    // Cluster charge and shape over the track's pixel hits only. Charge is in
                    // electrons; minChargeNorm is path-length normalised, Q*|sin(theta)| in the pixel
                    // barrel (detectorIndex < 864) and Q*|cos(theta)| in the endcap, and nLowCharge
                    // counts hits below 7000 e normalised. Cluster sizes are the raw signed
                    // 1/8-pixel values. All seven are -1 for a track with no usable pixel hit.
                    SOA_COLUMN(float, minCharge),
                    SOA_COLUMN(float, meanCharge),
                    SOA_COLUMN(float, minChargeNorm),
                    SOA_COLUMN(float, maxSizeY),
                    SOA_COLUMN(float, meanSizeY),
                    SOA_COLUMN(float, maxSizeX),
                    SOA_COLUMN(float, nLowCharge));

using PixelTrackFeaturesSoA = PixelTrackFeaturesSoALayout<>;

// Width of the feature vector the compact-forest scorer assembles positionally; the model loader
// validates every split-feature index against it.
inline constexpr int kNPixelTrackFeatures = 42;

// Define the SoA layout for track scores (output)
GENERATE_SOA_LAYOUT(PixelTrackScoresSoALayout, SOA_COLUMN(float, score))

using PixelTrackScoresSoA = PixelTrackScoresSoALayout<>;

#endif
