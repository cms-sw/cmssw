/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Mateusz Szpyrka (mateusz.szpyrka@cern.ch)
 *
 ****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/CTPPSDiamondTrackRecognition.h"

//----------------------------------------------------------------------------------------------------

CTPPSDiamondTrackRecognition::CTPPSDiamondTrackRecognition(const edm::ParameterSet& iConfig)
    : CTPPSTimingTrackRecognition<CTPPSDiamondLocalTrack, CTPPSDiamondRecHit>(iConfig),
      excludeSingleEdgeHits_(iConfig.getParameter<bool>("excludeSingleEdgeHits")) {}

//----------------------------------------------------------------------------------------------------

void CTPPSDiamondTrackRecognition::clear() {
  CTPPSTimingTrackRecognition<CTPPSDiamondLocalTrack, CTPPSDiamondRecHit>::clear();
  mhMap_.clear();
}

//----------------------------------------------------------------------------------------------------

void CTPPSDiamondTrackRecognition::addHit(const CTPPSDiamondRecHit& recHit) {
  if (excludeSingleEdgeHits_ && recHit.toT() <= 0.)
    return;
  // store hit parameters
  hitVectorMap_[recHit.ootIndex()].emplace_back(recHit);
}

//----------------------------------------------------------------------------------------------------

int CTPPSDiamondTrackRecognition::produceTracks(edm::DetSet<CTPPSDiamondLocalTrack>& tracks) {
  int numberOfTracks = 0;
  DimensionParameters param;

  auto getX = [](const CTPPSDiamondRecHit& hit) { return hit.x(); };
  auto getXWidth = [](const CTPPSDiamondRecHit& hit) { return hit.xWidth(); };
  auto setX = [](CTPPSDiamondLocalTrack& track, float x) { track.setPosition(math::XYZPoint(x, 0., 0.)); };
  auto setXSigma = [](CTPPSDiamondLocalTrack& track, float sigma) {
    track.setPositionSigma(math::XYZPoint(sigma, 0., 0.));
  };

  for (const auto& hitBatch : hitVectorMap_) {
    // separate the tracking for each bunch crossing
    const auto& oot = hitBatch.first;
    const auto& hits = hitBatch.second;

    auto hitRange = getHitSpatialRange(hits);

    TrackVector xPartTracks;

    // produce tracks in x dimension
    param.rangeBegin = hitRange.xBegin;
    param.rangeEnd = hitRange.xEnd;
    producePartialTracks(hits, param, getX, getXWidth, setX, setXSigma, xPartTracks);

    if (xPartTracks.empty())
      continue;

    const float yRangeCenter = 0.5f * (hitRange.yBegin + hitRange.yEnd);
    const float zRangeCenter = 0.5f * (hitRange.zBegin + hitRange.zEnd);
    const float ySigma = 0.5f * (hitRange.yEnd - hitRange.yBegin);
    const float zSigma = 0.5f * (hitRange.zEnd - hitRange.zBegin);

    for (const auto& xTrack : xPartTracks) {
      math::XYZPoint position(xTrack.x0(), yRangeCenter, zRangeCenter);
      math::XYZPoint positionSigma(xTrack.x0Sigma(), ySigma, zSigma);

      const int multipleHits = (mhMap_.find(oot) != mhMap_.end()) ? mhMap_[oot] : 0;
      CTPPSDiamondLocalTrack newTrack(position, positionSigma, 0.f, 0.f, oot, multipleHits);

      // find contributing hits
      HitVector componentHits;
      for (const auto& hit : hits)
        if (newTrack.containsHit(hit, tolerance_) && (!excludeSingleEdgeHits_ || hit.toT() > 0.))
          componentHits.emplace_back(hit);
      // compute timing information
      float mean_time = 0.f, time_sigma = 0.f;
      bool valid_hits = timeEval(componentHits, mean_time, time_sigma);
      newTrack.setValid(valid_hits);
      newTrack.setTime(mean_time);
      newTrack.setTimeSigma(time_sigma);

      tracks.push_back(newTrack);
    }
  }

  return numberOfTracks;
}
