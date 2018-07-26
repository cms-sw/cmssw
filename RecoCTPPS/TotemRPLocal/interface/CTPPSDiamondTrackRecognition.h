/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Mateusz Szpyrka (mateusz.szpyrka@cern.ch)
 *
 ****************************************************************************/

#ifndef RecoCTPPS_TotemRPLocal_CTPPSDiamondTrackRecognition
#define RecoCTPPS_TotemRPLocal_CTPPSDiamondTrackRecognition

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"

#include "RecoCTPPS/TotemRPLocal/interface/CTPPSTimingTrackRecognition.h"

#include <vector>
#include <unordered_map>
#include "TF1.h"

/**
 * \brief Class performing smart reconstruction for CTPPS Diamond Detectors.
 * \date Jan 2017
**/
class CTPPSDiamondTrackRecognition : public CTPPSTimingTrackRecognition<CTPPSDiamondLocalTrack, CTPPSDiamondRecHit>
{
  public:
    CTPPSDiamondTrackRecognition( const edm::ParameterSet& );

    void clear() override;

    /// Feed a new hit to the tracks recognition algorithm
    void addHit( const CTPPSDiamondRecHit& recHit ) override;

    /// Produce a collection of tracks for the current station, given its hits collection
    int produceTracks( edm::DetSet<CTPPSDiamondLocalTrack>& tracks ) override;

  private:

    std::unordered_map<int,int> mhMap_;
};



/****************************************************************************
 * Implementation
 ****************************************************************************/

//----------------------------------------------------------------------------------------------------

CTPPSDiamondTrackRecognition::CTPPSDiamondTrackRecognition(const edm::ParameterSet& parameters) :
    CTPPSTimingTrackRecognition<CTPPSDiamondLocalTrack, CTPPSDiamondRecHit>(parameters)
{}

//----------------------------------------------------------------------------------------------------

void CTPPSDiamondTrackRecognition::clear()
{
  CTPPSTimingTrackRecognition<CTPPSDiamondLocalTrack, CTPPSDiamondRecHit>::clear();
  mhMap_.clear();
}

//----------------------------------------------------------------------------------------------------

void CTPPSDiamondTrackRecognition::addHit( const CTPPSDiamondRecHit& recHit )
{
  // store hit parameters
  hitVectorMap[recHit.getOOTIndex()].emplace_back(recHit);
}

//----------------------------------------------------------------------------------------------------

int
CTPPSDiamondTrackRecognition::produceTracks(edm::DetSet<CTPPSDiamondLocalTrack>& tracks)
{
  int numberOfTracks = 0;
  DimensionParameters param;

  param.threshold = threshold;
  param.thresholdFromMaximum = thresholdFromMaximum;
  param.resolution = resolution;
  param.sigma = sigma;
  param.hitFunction = pixelEfficiencyFunction;


  for(auto hitBatch: hitVectorMap) {

    auto hits = hitBatch.second;
    auto oot = hitBatch.first;

    auto hitRange = getHitSpatialRange(hits);

    std::vector<CTPPSDiamondLocalTrack> xPartTracks;
    auto getX = [](const CTPPSDiamondRecHit& hit){ return hit.getX(); };
    auto getXWidth = [](const CTPPSDiamondRecHit& hit){ return hit.getXWidth(); };
    auto setX = [](CTPPSDiamondLocalTrack& track, float x){ track.setPosition(math::XYZPoint(x, 0., 0.)); };
    auto setXSigma = [](CTPPSDiamondLocalTrack& track, float sigma){ track.setPositionSigma(math::XYZPoint(sigma, 0., 0.)); };

    // Produces tracks in x dimension
    param.rangeBegin = hitRange.xBegin;
    param.rangeEnd = hitRange.xEnd;
    producePartialTracks(hits, param, getX, getXWidth, setX, setXSigma, xPartTracks);

    if(xPartTracks.empty())
      continue;

    float yRangeCenter = (hitRange.yBegin + hitRange.yEnd) / 2.0;
    float ySigma = (hitRange.yEnd - hitRange.yBegin) / 2.0;
    float zRangeCenter = (hitRange.zBegin + hitRange.zEnd) / 2.0;
    float zSigma = (hitRange.zEnd - hitRange.zBegin) / 2.0;

    for(const auto& xTrack: xPartTracks) {
      math::XYZPoint position(
        xTrack.getX0(),
        yRangeCenter,
        zRangeCenter
      );
      math::XYZPoint positionSigma(
        xTrack.getX0Sigma(),
        ySigma,
        zSigma
      );

      CTPPSDiamondLocalTrack newTrack;
      newTrack.setPosition(position);
      newTrack.setPositionSigma(positionSigma);
      newTrack.setValid(true);
      newTrack.setOOTIndex(oot);
      int multipleHits = (mhMap_.find(oot) != mhMap_.end()) ? mhMap_[oot] : 0;
      newTrack.setMultipleHits(multipleHits);

      tracks.push_back(newTrack);
    }
  }

  return numberOfTracks;
}

#endif
