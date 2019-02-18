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

CTPPSDiamondTrackRecognition::CTPPSDiamondTrackRecognition( const edm::ParameterSet& iConfig ) :
  CTPPSTimingTrackRecognition<CTPPSDiamondLocalTrack, CTPPSDiamondRecHit>( iConfig )
{}

//----------------------------------------------------------------------------------------------------

void
CTPPSDiamondTrackRecognition::clear()
{
  CTPPSTimingTrackRecognition<CTPPSDiamondLocalTrack, CTPPSDiamondRecHit>::clear();
  mhMap_.clear();
}

//----------------------------------------------------------------------------------------------------

void
CTPPSDiamondTrackRecognition::addHit( const CTPPSDiamondRecHit& recHit )
{
  // store hit parameters
  hitVectorMap_[recHit.getOOTIndex()].emplace_back( recHit );
}

//----------------------------------------------------------------------------------------------------

int
CTPPSDiamondTrackRecognition::produceTracks( edm::DetSet<CTPPSDiamondLocalTrack>& tracks )
{
  int numberOfTracks = 0;
  DimensionParameters param;

  auto getX = []( const CTPPSDiamondRecHit& hit ){ return hit.getX(); };
  auto getXWidth = []( const CTPPSDiamondRecHit& hit ){ return hit.getXWidth(); };
  auto setX = []( CTPPSDiamondLocalTrack& track, float x ){ track.setPosition( math::XYZPoint( x, 0., 0. ) ); };
  auto setXSigma = []( CTPPSDiamondLocalTrack& track, float sigma ){ track.setPositionSigma( math::XYZPoint( sigma, 0., 0. ) ); };

  for ( const auto& hitBatch: hitVectorMap_ ) {
    const auto& oot = hitBatch.first;
    const auto& hits = hitBatch.second;

    auto hitRange = getHitSpatialRange( hits );

    std::vector<CTPPSDiamondLocalTrack> xPartTracks;

    // Produces tracks in x dimension
    param.rangeBegin = hitRange.xBegin;
    param.rangeEnd = hitRange.xEnd;
    producePartialTracks( hits, param, getX, getXWidth, setX, setXSigma, xPartTracks );

    if ( xPartTracks.empty() )
      continue;

    const float yRangeCenter = 0.5f*( hitRange.yBegin + hitRange.yEnd );
    const float zRangeCenter = 0.5f*( hitRange.zBegin + hitRange.zEnd );
    const float ySigma = 0.5f*( hitRange.yEnd - hitRange.yBegin );
    const float zSigma = 0.5f*( hitRange.zEnd - hitRange.zBegin );

    for ( const auto& xTrack: xPartTracks ) {
      math::XYZPoint position( xTrack.getX0(), yRangeCenter, zRangeCenter );
      math::XYZPoint positionSigma( xTrack.getX0Sigma(), ySigma, zSigma );

      const int multipleHits = ( mhMap_.find(oot) != mhMap_.end() )
        ? mhMap_[oot]
        : 0;
      CTPPSDiamondLocalTrack newTrack( position, positionSigma, 0.f, 0.f, oot, multipleHits );
      newTrack.setValid( true );

      tracks.push_back( newTrack );
    }
  }

  return numberOfTracks;
}

