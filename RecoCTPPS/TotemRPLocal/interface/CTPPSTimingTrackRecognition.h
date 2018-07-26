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

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSReco/interface/CTPPSTimingRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSTimingLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingLocalTrack.h"

#include <string>
#include <cmath>
#include <set>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "TF1.h"

/**
 * Class intended to perform general CTPPS timing detectors track recognition,
 * as well as construction of specialized classes (for now CTPPSDiamond and TotemTiming local tracks).
**/
template<typename TRACK_TYPE, typename HIT_TYPE>
class CTPPSTimingTrackRecognition
{
  public:

    CTPPSTimingTrackRecognition(const edm::ParameterSet& parameters);

    virtual ~CTPPSTimingTrackRecognition();


    // Class API:

    // Resets internal state of a class instance.
    virtual void clear();


    // Adds new hit to the set from which the tracks are reconstructed.
    virtual void addHit(const HIT_TYPE& recHit) = 0;


    // Produces a collection of tracks, given its hits collection
    virtual int produceTracks(edm::DetSet<TRACK_TYPE>& tracks) = 0;



  protected:

    // Algorithm parameters:
    const float threshold;
    const float thresholdFromMaximum;
    const float resolution;
    const float sigma;
    TF1 pixelEfficiencyFunction;


    typedef std::vector<HIT_TYPE> HitVector;
    typedef std::unordered_map<int, HitVector> HitVectorMap;


    /* Stores RecHit vectors that should be processed separately while reconstructing tracks. */
    HitVectorMap hitVectorMap;


    /* Structure representing parameters set for single dimension.
     * Intended to use when producing partial tracks.
     */
    struct DimensionParameters {
      float threshold;
      float thresholdFromMaximum;
      float resolution;
      float sigma;
      float rangeBegin;
      float rangeEnd;
      TF1 hitFunction;
    };


    /* Structure representing a 3D range in space.
     */
    struct SpatialRange {
      float xBegin;
      float xEnd;
      float yBegin;
      float yEnd;
      float zBegin;
      float zEnd;
    };


    /* Produces all partial tracks from given set with regard to single dimension.
     *
     * @hits: vector of hits from which the tracks are created
     * @param: describes all parameters used by 1D track recognition algorithm
     * @getHitCenter: function extracting hit's center in the dimension
     *      that the partial tracks relate to
     * @getHitRangeWidth: analogical to getHitCenter, but extracts hit's width
     *      in specific dimension
     * @setTrackCenter: function used to set track's position in considered dimension
     * @setTrackSigma: function used to set track's sigma in considered dimension
     * @result: vector to which produced tracks are appended
     */
    void producePartialTracks(
        const HitVector& hits,
        const DimensionParameters& param,
        float (*getHitCenter)(const HIT_TYPE&),
        float (*getHitRangeWidth)(const HIT_TYPE&),
        void (*setTrackCenter)(TRACK_TYPE&, float),
        void (*setTrackSigma)(TRACK_TYPE&, float),
        std::vector<TRACK_TYPE>& result
      );



    /* Retrieves the bounds of a 3D range in which all hits from given collection are contained.
     *
     * @hits: hits collection to retrieve the range from
     */
    SpatialRange getHitSpatialRange(const HitVector& hits);
};

/****************************************************************************
 * Implementation
 ****************************************************************************/


template<class TRACK_TYPE, class HIT_TYPE>
CTPPSTimingTrackRecognition<TRACK_TYPE, HIT_TYPE>::CTPPSTimingTrackRecognition(const edm::ParameterSet& iConfig) :
    threshold               ( iConfig.getParameter<double>( "threshold" ) ),
    thresholdFromMaximum    ( iConfig.getParameter<double>( "thresholdFromMaximum" ) ),
    resolution              ( iConfig.getParameter<double>( "resolution" ) ),
    sigma                   ( iConfig.getParameter<double>( "sigma" ) ),
    pixelEfficiencyFunction ( "hit_TF1_CTPPS", iConfig.getParameter<std::string>( "pixelEfficiencyFunction" ).c_str() ) {
}


template<class TRACK_TYPE, class HIT_TYPE>
CTPPSTimingTrackRecognition<TRACK_TYPE, HIT_TYPE>::~CTPPSTimingTrackRecognition() {};


template<class TRACK_TYPE, class HIT_TYPE>
void CTPPSTimingTrackRecognition<TRACK_TYPE, HIT_TYPE>::clear() {
  hitVectorMap.clear();
}


template<class TRACK_TYPE, class HIT_TYPE>
void CTPPSTimingTrackRecognition<TRACK_TYPE, HIT_TYPE>::producePartialTracks(
    const HitVector& hits,
    const DimensionParameters& param,
    float (*getHitCenter)(const HIT_TYPE&),
    float (*getHitRangeWidth)(const HIT_TYPE&),
    void (*setTrackCenter)(TRACK_TYPE&, float),
    void (*setTrackSigma)(TRACK_TYPE&, float),
    std::vector<TRACK_TYPE>& result
  ) {

  int numberOfTracks = 0;
  const double invResolution = 1./param.resolution;
  const float profileRangeMargin = param.sigma * 3.;
  const float profileRangeBegin = param.rangeBegin - profileRangeMargin;
  const float profileRangeEnd = param.rangeEnd + profileRangeMargin;

  std::vector<float> hitProfile((profileRangeEnd - profileRangeBegin) * invResolution, 0.);
  auto hitFunction = param.hitFunction;

  // Creates hit profile
  for(auto const& hit : hits) {

    float center = getHitCenter(hit);
    float rangeWidth = getHitRangeWidth(hit);

    hitFunction.SetParameters(center, rangeWidth, param.sigma);

    for(unsigned int i = 0; i < hitProfile.size(); ++i) {
      hitProfile[i] += hitFunction.Eval(profileRangeBegin + i*param.resolution);
    }
  }

  // Guard to make sure that the profile drops below the threshold at range's end
  hitProfile.push_back(-1.);

  bool underThreshold = true;
  float rangeMaximum = -1.0;
  bool trackRangeFound = false;
  int trackRangeBegin = 0;
  int trackRangeEnd;

  // Searches for tracks in the hit profile
  for(unsigned int i = 0; i < hitProfile.size(); i++) {

    if(hitProfile[i] > rangeMaximum)
      rangeMaximum = hitProfile[i];

    // Going above the threshold
    if(underThreshold && hitProfile[i] > param.threshold) {
      underThreshold = false;
      trackRangeBegin = i;
      rangeMaximum = hitProfile[i];
    }

    // Going under the threshold
    else if(!underThreshold && hitProfile[i] <= param.threshold){
      underThreshold = true;
      trackRangeEnd = i;
      trackRangeFound = true;
    }

    // Finds all tracks within the track range
    if(trackRangeFound) {

      float trackThreshold = rangeMaximum - param.thresholdFromMaximum;
      int trackBegin;
      bool underTrackThreshold = true;

      for(int j = trackRangeBegin; j <= trackRangeEnd; j++) {

        if(underTrackThreshold && hitProfile[j] > trackThreshold) {
          underTrackThreshold = false;
          trackBegin = j;
        }

        else if(!underTrackThreshold && hitProfile[j] <= trackThreshold) {
          underTrackThreshold = true;
          TRACK_TYPE track;
          float leftMargin = profileRangeBegin + param.resolution * trackBegin;
          float rightMargin = profileRangeBegin + param.resolution * j;
          setTrackCenter(track, (leftMargin + rightMargin) / 2.0);
          setTrackSigma(track, (rightMargin - leftMargin) / 2.0);
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
CTPPSTimingTrackRecognition<TRACK_TYPE, HIT_TYPE>::getHitSpatialRange(const HitVector& hits) {

  bool initialized = false;
  SpatialRange result;

  for(const auto& hit: hits) {

    if(initialized) {
      float xBegin = hit.getX() - (hit.getXWidth() / 2.0);
      float xEnd = hit.getX() + (hit.getXWidth() / 2.0);
      float yBegin = hit.getY() - (hit.getYWidth() / 2.0);
      float yEnd = hit.getY() + (hit.getYWidth() / 2.0);
      float zBegin = hit.getZ() - (hit.getZWidth() / 2.0);
      float zEnd = hit.getZ() + (hit.getZWidth() / 2.0);

      if(xBegin < result.xBegin)
        result.xBegin = xBegin;

      if(xEnd > result.xEnd)
        result.xEnd = xEnd;

      if(yBegin < result.yBegin)
        result.yBegin = yBegin;

      if(yEnd > result.yEnd)
        result.yEnd = yEnd;

      if(zBegin < result.zBegin)
        result.zBegin = zBegin;

      if(zEnd > result.zEnd)
        result.zEnd = zEnd;
    }

    else {
      result.xBegin = hit.getX() - (hit.getXWidth() / 2.0);
      result.xEnd = hit.getX() + (hit.getXWidth() / 2.0);
      result.yBegin = hit.getY() - (hit.getYWidth() / 2.0);
      result.yEnd = hit.getY() + (hit.getYWidth() / 2.0);
      result.zBegin = hit.getZ() - (hit.getZWidth() / 2.0);
      result.zEnd = hit.getZ() + (hit.getZWidth() / 2.0);
      initialized = true;
    }
  }

  return result;
}

#endif
