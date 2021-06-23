#ifndef RecoBTag_FeatureTools_SeedingTracksConverter_h
#define RecoBTag_FeatureTools_SeedingTracksConverter_h

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/BTauReco/interface/SeedingTrackFeatures.h"
#include "DataFormats/BTauReco/interface/TrackPairFeatures.h"

#include "RecoBTag/FeatureTools/interface/TrackPairInfoBuilder.h"
#include "RecoBTag/FeatureTools/interface/SeedingTrackInfoBuilder.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"

#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"

namespace btagbtvdeep {

  void seedingTracksToFeatures(const std::vector<reco::TransientTrack>& selectedTracks,
                               const std::vector<float>& masses,
                               const reco::Jet& jet,
                               const reco::Vertex& pv,
                               HistogramProbabilityEstimator* probabilityEstimator,
                               bool computeProbabilities,
                               std::vector<btagbtvdeep::SeedingTrackFeatures>& seedingT_features_vector);

  inline float logWithOffset(float v, float logOffset = 0) {
    if (v == 0.)
      return 0.;
    return logOffset + log(std::fabs(v)) * std::copysign(1.f, v);
  };
}  // namespace btagbtvdeep

#endif  //RecoBTag_FeatureTools_SeedingTracksConverter_h
