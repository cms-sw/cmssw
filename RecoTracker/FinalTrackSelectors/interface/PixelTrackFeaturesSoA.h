#ifndef RecoTracker_FinalTrackSelectors_PixelTrackFeaturesSoA_h
#define RecoTracker_FinalTrackSelectors_PixelTrackFeaturesSoA_h

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

// Per-event scratch of the high-purity selectors, never an event product: each selector allocates
// one per event, fills it in its extractor kernel and scores it in the same produce(). It is a
// two-block layout: the fit block (17 fit/covariance columns) is always allocated and written for
// every preselected track, the hit block (25 CA hit/stub, merged-collection provenance and pixel
// cluster columns) is allocated and written only when useHitFeatures is true, i.e. only when a
// model consumes it. Both selectors read the columns positionally in this order (a model reads the
// first N columns it was trained on), so columns are only ever appended.
GENERATE_SOA_LAYOUT(PixelTrackFitFeaturesLayout,
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
                    SOA_COLUMN(float, covPhiQOverPt));

GENERATE_SOA_LAYOUT(PixelTrackHitFeaturesLayout,
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

GENERATE_SOA_BLOCKS(PixelTrackFeaturesBlocksLayout,
                    SOA_BLOCK(fit, PixelTrackFitFeaturesLayout),
                    SOA_BLOCK(hit, PixelTrackHitFeaturesLayout))

using PixelTrackFitFeaturesSoA = PixelTrackFitFeaturesLayout<>;
using PixelTrackFitFeaturesView = PixelTrackFitFeaturesSoA::View;
using PixelTrackFitFeaturesConstView = PixelTrackFitFeaturesSoA::ConstView;

using PixelTrackHitFeaturesSoA = PixelTrackHitFeaturesLayout<>;
using PixelTrackHitFeaturesView = PixelTrackHitFeaturesSoA::View;
using PixelTrackHitFeaturesConstView = PixelTrackHitFeaturesSoA::ConstView;

using PixelTrackFeaturesBlocksSoA = PixelTrackFeaturesBlocksLayout<>;

// Columns of an all-float layout: with one alignment-wide bunch of elements each column takes
// one alignment unit.
template <typename TLayout>
inline constexpr int nFloatColumns = TLayout::computeDataSize(TLayout::alignment / sizeof(float)) / TLayout::alignment;

inline constexpr int kNPixelTrackFitFeatures = nFloatColumns<PixelTrackFitFeaturesSoA>;
inline constexpr int kNPixelTrackHitFeatures = nFloatColumns<PixelTrackHitFeaturesSoA>;

// Width of the feature vector the compact-forest scorer assembles positionally out of the two
// blocks; the model loader validates every split-feature index against it.
inline constexpr int kNPixelTrackFeatures = 42;
static_assert(kNPixelTrackFeatures == kNPixelTrackFitFeatures + kNPixelTrackHitFeatures);

// Define the SoA layout for track scores (output)
GENERATE_SOA_LAYOUT(PixelTrackScoresSoALayout, SOA_COLUMN(float, score))

using PixelTrackScoresSoA = PixelTrackScoresSoALayout<>;

#endif
