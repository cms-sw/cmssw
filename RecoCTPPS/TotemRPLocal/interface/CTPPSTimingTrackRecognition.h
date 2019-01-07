/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Mateusz Szpyrka (mateusz.szpyrka@cern.ch)
 *
 ****************************************************************************/

#ifndef RecoCTPPS_TotemRPLocal_CTPPSTimingTrackRecognition
#define RecoCTPPS_TotemRPLocal_CTPPSTimingTrackRecognition

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "DataFormats/Common/interface/DetSet.h"

#include <vector>
#include <unordered_map>
#include <algorithm>

/**
 * Class intended to perform general CTPPS timing detectors track recognition,
 * as well as construction of specialized classes (for now CTPPSDiamond and TotemTiming local tracks).
**/
template<typename TRACK_TYPE, typename HIT_TYPE>
class CTPPSTimingTrackRecognition
{
  public:
    CTPPSTimingTrackRecognition(const edm::ParameterSet& iConfig) :
      threshold_              (iConfig.getParameter<double>("threshold")),
      thresholdFromMaximum_   (iConfig.getParameter<double>("thresholdFromMaximum")),
      resolution_             (iConfig.getParameter<double>("resolution")),
      sigma_                  (iConfig.getParameter<double>("sigma")),
      pixelEfficiencyFunction_(iConfig.getParameter<std::string>("pixelEfficiencyFunction")) {}
    virtual ~CTPPSTimingTrackRecognition() = default;

    //--- class API

    /// Reset internal state of a class instance.
    virtual void clear() { hitVectorMap_.clear(); }
    /// Add new hit to the set from which the tracks are reconstructed.
    virtual void addHit(const HIT_TYPE& recHit) = 0;
    /// Produce a collection of tracks, given its hits collection
    virtual int produceTracks(edm::DetSet<TRACK_TYPE>& tracks) = 0;

  protected:
    // Algorithm parameters:
    const float threshold_;
    const float thresholdFromMaximum_;
    const float resolution_;
    const float sigma_;
    reco::FormulaEvaluator pixelEfficiencyFunction_;

    typedef std::vector<HIT_TYPE> HitVector;
    typedef std::unordered_map<int, HitVector> HitVectorMap;

    /// RecHit vectors that should be processed separately while reconstructing tracks.
    HitVectorMap hitVectorMap_;

    /// Structure representing parameters set for single dimension.
    /// Intended to use when producing partial tracks.
    struct DimensionParameters { float rangeBegin, rangeEnd; };
    /// Structure representing a 3D range in space.
    struct SpatialRange {
      float xBegin, xEnd;
      float yBegin, yEnd;
      float zBegin, zEnd;
    };

    /** Produce all partial tracks from given set with regard to single dimension.
     * \param[in] hits vector of hits from which the tracks are created
     * \param[in] param describe all parameters used by 1D track recognition algorithm
     * \param[in] getHitCenter function extracting hit's center in the dimension that the partial tracks relate to
     * \param[in] getHitRangeWidth analogue to \a getHitCenter, but extracts hit's width in specific dimension
     * \param[in] setTrackCenter function used to set track's position in considered dimension
     * \param[in] setTrackSigma function used to set track's sigma in considered dimension
     * \param[out] result vector to which produced tracks are appended
     */
    void producePartialTracks(
        const HitVector& hits,
        const DimensionParameters& param,
        float (*getHitCenter)(const HIT_TYPE&),
        float (*getHitRangeWidth)(const HIT_TYPE&),
        void (*setTrackCenter)(TRACK_TYPE&, float),
        void (*setTrackSigma)(TRACK_TYPE&, float),
        std::vector<TRACK_TYPE>& result);

    /** Retrieve the bounds of a 3D range in which all hits from given collection are contained.
     * \param[in] hits hits collection to retrieve the range from
     */
    SpatialRange getHitSpatialRange(const HitVector& hits);
};

/****************************************************************************
 * Implementation
 ****************************************************************************/

template<class TRACK_TYPE, class HIT_TYPE>
void CTPPSTimingTrackRecognition<TRACK_TYPE, HIT_TYPE>::producePartialTracks(
    const HitVector& hits,
    const DimensionParameters& param,
    float (*getHitCenter)(const HIT_TYPE&),
    float (*getHitRangeWidth)(const HIT_TYPE&),
    void (*setTrackCenter)(TRACK_TYPE&, float),
    void (*setTrackSigma)(TRACK_TYPE&, float),
    std::vector<TRACK_TYPE>& result) {
  int numberOfTracks = 0;
  const float invResolution = 1./resolution_;
  const float profileRangeMargin = sigma_ * 3.;
  const float profileRangeBegin = param.rangeBegin - profileRangeMargin;
  const float profileRangeEnd = param.rangeEnd + profileRangeMargin;

  std::vector<float> hitProfile((profileRangeEnd - profileRangeBegin) * invResolution + 1, 0.);
  // extra component to make sure that the profile drops below the threshold at range's end
  *hitProfile.rbegin() = -1.f;

  // Creates hit profile
  for (auto const& hit : hits) {
    const float center = getHitCenter(hit), rangeWidth = getHitRangeWidth(hit);
    std::vector<double> params{center, rangeWidth, sigma_};
    for (unsigned int i = 0; i < hitProfile.size(); ++i)
      hitProfile[i] += pixelEfficiencyFunction_.evaluate(std::vector<double>{profileRangeBegin + i*resolution_}, params);
  }

  bool underThreshold = true;
  float rangeMaximum = -1.0f;
  bool trackRangeFound = false;
  int trackRangeBegin = 0;
  int trackRangeEnd;

  // Searches for tracks in the hit profile
  for (unsigned int i = 0; i < hitProfile.size(); i++) {
    if (hitProfile[i] > rangeMaximum)
      rangeMaximum = hitProfile[i];

    // Going above the threshold
    if (underThreshold && hitProfile[i] > threshold_) {
      underThreshold = false;
      trackRangeBegin = i;
      rangeMaximum = hitProfile[i];
    }

    // Going under the threshold
    else if (!underThreshold && hitProfile[i] <= threshold_) {
      underThreshold = true;
      trackRangeEnd = i;
      trackRangeFound = true;
    }

    // Finds all tracks within the track range
    if (trackRangeFound) {
      float trackThreshold = rangeMaximum - thresholdFromMaximum_;
      int trackBegin;
      bool underTrackThreshold = true;

      for (int j = trackRangeBegin; j <= trackRangeEnd; j++) {
        if (underTrackThreshold && hitProfile[j] > trackThreshold) {
          underTrackThreshold = false;
          trackBegin = j;
        }
        else if (!underTrackThreshold && hitProfile[j] <= trackThreshold) {
          underTrackThreshold = true;
          TRACK_TYPE track;
          float leftMargin = profileRangeBegin + resolution_ * trackBegin;
          float rightMargin = profileRangeBegin + resolution_ * j;
          setTrackCenter(track, 0.5f*(leftMargin + rightMargin));
          setTrackSigma(track, 0.5f*(rightMargin - leftMargin));
          result.push_back(track);
          numberOfTracks++;
        }
      }
      trackRangeFound = false;
    }
  }
}

template<class TRACK_TYPE, class HIT_TYPE>
typename CTPPSTimingTrackRecognition<TRACK_TYPE, HIT_TYPE>::SpatialRange
CTPPSTimingTrackRecognition<TRACK_TYPE, HIT_TYPE>::getHitSpatialRange(const HitVector& hits)
{
  bool initialized = false;
  SpatialRange result;

  for (const auto& hit : hits) {
    const float xBegin = hit.getX() - 0.5f*hit.getXWidth();
    const float yBegin = hit.getY() - 0.5f*hit.getYWidth();
    const float zBegin = hit.getZ() - 0.5f*hit.getZWidth();
    const float xEnd = hit.getX() + 0.5f*hit.getXWidth();
    const float yEnd = hit.getY() + 0.5f*hit.getYWidth();
    const float zEnd = hit.getZ() + 0.5f*hit.getZWidth();

    if (!initialized) {
      result.xBegin = xBegin;
      result.yBegin = yBegin;
      result.zBegin = zBegin;
      result.xEnd = xEnd;
      result.yEnd = yEnd;
      result.zEnd = zEnd;
      initialized = true;
      continue;
    }
    if (xBegin < result.xBegin) result.xBegin = xBegin;
    if (yBegin < result.yBegin) result.yBegin = yBegin;
    if (zBegin < result.zBegin) result.zBegin = zBegin;

    if (xEnd > result.xEnd) result.xEnd = xEnd;
    if (yEnd > result.yEnd) result.yEnd = yEnd;
    if (zEnd > result.zEnd) result.zEnd = zEnd;
  }

  return result;
}

#endif

