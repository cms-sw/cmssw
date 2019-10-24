/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Mateusz Szpyrka (mateusz.szpyrka@cern.ch)
 *
 ****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/TotemTimingTrackRecognition.h"

//----------------------------------------------------------------------------------------------------

TotemTimingTrackRecognition::TotemTimingTrackRecognition(const edm::ParameterSet& iConfig)
    : CTPPSTimingTrackRecognition<TotemTimingLocalTrack, TotemTimingRecHit>(iConfig) {}

//----------------------------------------------------------------------------------------------------

void TotemTimingTrackRecognition::addHit(const TotemTimingRecHit& recHit) {
  if (recHit.time() != TotemTimingRecHit::NO_T_AVAILABLE)
    hitVectorMap_[0].emplace_back(recHit);
}

//----------------------------------------------------------------------------------------------------

int TotemTimingTrackRecognition::produceTracks(edm::DetSet<TotemTimingLocalTrack>& tracks) {
  int numberOfTracks = 0;
  DimensionParameters param;

  auto getX = [](const TotemTimingRecHit& hit) { return hit.x(); };
  auto getXWidth = [](const TotemTimingRecHit& hit) { return hit.xWidth(); };
  auto setX = [](TotemTimingLocalTrack& track, float x) { track.setPosition(math::XYZPoint(x, 0., 0.)); };
  auto setXSigma = [](TotemTimingLocalTrack& track, float sigma) {
    track.setPositionSigma(math::XYZPoint(sigma, 0., 0.));
  };
  auto getY = [](const TotemTimingRecHit& hit) { return hit.y(); };
  auto getYWidth = [](const TotemTimingRecHit& hit) { return hit.yWidth(); };
  auto setY = [](TotemTimingLocalTrack& track, float y) { track.setPosition(math::XYZPoint(0., y, 0.)); };
  auto setYSigma = [](TotemTimingLocalTrack& track, float sigma) {
    track.setPositionSigma(math::XYZPoint(0., sigma, 0.));
  };

  for (const auto& hitBatch : hitVectorMap_) {
    const auto& hits = hitBatch.second;
    const auto& hitRange = getHitSpatialRange(hits);

    TrackVector xPartTracks, yPartTracks;

    param.rangeBegin = hitRange.xBegin;
    param.rangeEnd = hitRange.xEnd;
    producePartialTracks(hits, param, getX, getXWidth, setX, setXSigma, xPartTracks);

    param.rangeBegin = hitRange.yBegin;
    param.rangeEnd = hitRange.yEnd;
    producePartialTracks(hits, param, getY, getYWidth, setY, setYSigma, yPartTracks);

    if (xPartTracks.empty() && yPartTracks.empty())
      continue;

    unsigned int validHitsNumber = (unsigned int)threshold_ + 1;

    for (const auto& xTrack : xPartTracks) {
      for (const auto& yTrack : yPartTracks) {
        math::XYZPoint position(xTrack.x0(), yTrack.y0(), 0.5f * (hitRange.zBegin + hitRange.zEnd));
        math::XYZPoint positionSigma(xTrack.x0Sigma(), yTrack.y0Sigma(), 0.5f * (hitRange.zEnd - hitRange.zBegin));

        TotemTimingLocalTrack newTrack(position, positionSigma, 0., 0.);

        HitVector componentHits;
        for (const auto& hit : hits)
          if (newTrack.containsHit(hit, tolerance_))
            componentHits.emplace_back(hit);
        if (componentHits.size() < validHitsNumber)
          continue;

        float mean_time = 0.f, time_sigma = 0.f;
        bool valid_hits = timeEval(componentHits, mean_time, time_sigma);
        newTrack.setValid(valid_hits);
        newTrack.setTime(mean_time);
        newTrack.setTimeSigma(time_sigma);
        // in a next iteration, we will be setting validity / numHits / numPlanes
        tracks.push_back(newTrack);
      }
    }
  }

  return numberOfTracks;
}
