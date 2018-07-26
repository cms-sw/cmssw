/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Mateusz Szpyrka (mateusz.szpyrka@cern.ch)
 *
 ****************************************************************************/

#ifndef RecoCTPPS_TotemRPLocal_TotemTimingTrackRecognition
#define RecoCTPPS_TotemRPLocal_TotemTimingTrackRecognition

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSReco/interface/CTPPSTimingRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSTimingLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingLocalTrack.h"
#include "RecoCTPPS/TotemRPLocal/interface/CTPPSTimingTrackRecognition.h"


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

class TotemTimingTrackRecognition : public CTPPSTimingTrackRecognition<TotemTimingLocalTrack, TotemTimingRecHit>
{
  public:

    TotemTimingTrackRecognition(const edm::ParameterSet& parameters);

    // Adds new hit to the set from which the tracks are reconstructed.
    void addHit(const TotemTimingRecHit& recHit) override;

    /// Produces a collection of tracks for the current station, given its hits collection
    int produceTracks(edm::DetSet<TotemTimingLocalTrack>& tracks) override;

  private:

    float tolerance;
};


/****************************************************************************
 * Implementation
 ****************************************************************************/

TotemTimingTrackRecognition::TotemTimingTrackRecognition(const edm::ParameterSet& parameters) :
    CTPPSTimingTrackRecognition<TotemTimingLocalTrack, TotemTimingRecHit>(parameters),
    tolerance(parameters.getParameter<double>( "tolerance" ))
  {};

void TotemTimingTrackRecognition::addHit(const TotemTimingRecHit& recHit) {
  if(recHit.getT() != TotemTimingRecHit::NO_T_AVAILABLE)
    hitVectorMap[0].push_back(recHit);
}

int TotemTimingTrackRecognition::produceTracks(edm::DetSet<TotemTimingLocalTrack>& tracks) {

  int numberOfTracks = 0;
  DimensionParameters param;

  param.threshold = threshold;
  param.thresholdFromMaximum = thresholdFromMaximum;
  param.resolution = resolution;
  param.sigma = sigma;
  param.hitFunction = pixelEfficiencyFunction;

  for(auto hitBatch: hitVectorMap) {

    auto hits = hitBatch.second;
    auto hitRange = getHitSpatialRange(hits);

    std::vector<TotemTimingLocalTrack> xPartTracks, yPartTracks;
    auto getX = [](const TotemTimingRecHit& hit){ return hit.getX(); };
    auto getXWidth = [](const TotemTimingRecHit& hit){ return hit.getXWidth(); };
    auto setX = [](TotemTimingLocalTrack& track, float x){ track.setPosition(math::XYZPoint(x, 0., 0.)); };
    auto setXSigma = [](TotemTimingLocalTrack& track, float sigma){ track.setPositionSigma(math::XYZPoint(sigma, 0., 0.)); };
    auto getY = [](const TotemTimingRecHit& hit){ return hit.getY(); };
    auto getYWidth = [](const TotemTimingRecHit& hit){ return hit.getYWidth(); };
    auto setY = [](TotemTimingLocalTrack& track, float y){ track.setPosition(math::XYZPoint(0., y, 0.)); };
    auto setYSigma = [](TotemTimingLocalTrack& track, float sigma){ track.setPositionSigma(math::XYZPoint(0., sigma, 0.)); };

    param.rangeBegin = hitRange.xBegin;
    param.rangeEnd = hitRange.xEnd;
    producePartialTracks(hits, param, getX, getXWidth, setX, setXSigma, xPartTracks);

    param.rangeBegin = hitRange.yBegin;
    param.rangeEnd = hitRange.yEnd;
    producePartialTracks(hits, param, getY, getYWidth, setY, setYSigma, yPartTracks);

    if(xPartTracks.empty() && yPartTracks.empty())
     continue;

    unsigned int validHitsNumber = (unsigned int)(threshold + 1.0);

    for(const auto& xTrack: xPartTracks) {
      for(const auto& yTrack: yPartTracks) {

        math::XYZPoint position(
          xTrack.getX0(),
          yTrack.getY0(),
          (hitRange.zBegin + hitRange.zEnd) / 2.0
        );
        math::XYZPoint positionSigma(
          xTrack.getX0Sigma(),
          yTrack.getY0Sigma(),
          (hitRange.zEnd - hitRange.zBegin) / 2.0
        );

        TotemTimingLocalTrack newTrack;
        newTrack.setPosition(position);
        newTrack.setPositionSigma(positionSigma);

        std::vector<TotemTimingRecHit> componentHits;
        for(auto hit: hits)
          if(newTrack.containsHit(hit, tolerance))
            componentHits.push_back(hit);

        if(componentHits.size() < validHitsNumber)
          continue;

        // Calculating time
        //    track's time = weighted mean of all hit times whith time precision as weight
        //    track's time sigma = uncertainty of the weighted mean
        // hit is ignored if the time precision is equal to 0

        float meanDivident = 0.;
        float meanDivisor = 0.;
        bool validHits = false;
        for(const auto& hit : componentHits) {

          if(hit.getTPrecision() == 0.)
            continue;

          validHits = true;
          float weight = 1 / (hit.getTPrecision() * hit.getTPrecision());
          meanDivident += weight * hit.getT();
          meanDivisor += weight;
        }

        float meanTime = validHits ? (meanDivident / meanDivisor) : 0.;
        float timeSigma = validHits ? (std::sqrt(1 / meanDivisor)) : 0.;
        newTrack.setValid(validHits);
        newTrack.setT(meanTime);
        newTrack.setTSigma(timeSigma);
        // TODO: setting validity / numHits / numPlanes
        tracks.push_back(newTrack);
      }
    }
  }

  return numberOfTracks;
}

#endif
