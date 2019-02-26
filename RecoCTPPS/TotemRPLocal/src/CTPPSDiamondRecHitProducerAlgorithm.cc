/****************************************************************************
*
* This is a part of CTPPS offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/CTPPSDiamondRecHitProducerAlgorithm.h"

//----------------------------------------------------------------------------------------------------

CTPPSDiamondRecHitProducerAlgorithm::CTPPSDiamondRecHitProducerAlgorithm( const edm::ParameterSet& iConfig ) :
  ts_to_ns_( iConfig.getParameter<double>( "timeSliceNs" ) )
{}

void
CTPPSDiamondRecHitProducerAlgorithm::build( const CTPPSGeometry* geom, const edm::DetSetVector<CTPPSDiamondDigi>& input, edm::DetSetVector<CTPPSDiamondRecHit>& output )
{
  for ( const auto& vec : input ) {
    const CTPPSDiamondDetId detid( vec.detId() );

    if ( detid.channel() > 20 ) continue; // VFAT-like information, to be ignored

    // retrieve the geometry element associated to this DetID
    const DetGeomDesc* det = geom->getSensor( detid );

    const float x_pos = det->translation().x(),
                y_pos = det->translation().y();
    float z_pos = 0.;
    z_pos = det->parentZPosition(); // retrieve the plane position;

    const float x_width = 2.0 * det->params().at( 0 ), // parameters stand for half the size
                y_width = 2.0 * det->params().at( 1 ),
                z_width = 2.0 * det->params().at( 2 );

    edm::DetSet<CTPPSDiamondRecHit>& rec_hits = output.find_or_insert( detid );

    for ( const auto& digi : vec ) {
      if ( digi.getLeadingEdge() == 0 && digi.getTrailingEdge() == 0 ) continue;

      const int t = digi.getLeadingEdge();
      const int t0 = t % 1024;
      const int time_slice = ( t != 0 ) ? t / 1024 : CTPPSDiamondRecHit::TIMESLICE_WITHOUT_LEADING;

      int tot = 0;
      if ( t != 0 && digi.getTrailingEdge() != 0 ) tot = ( (int)digi.getTrailingEdge() ) - t;

      rec_hits.push_back( CTPPSDiamondRecHit( x_pos, x_width, y_pos, y_width, z_pos, z_width, // spatial information
                                              ( t0 * ts_to_ns_ ),
                                              ( tot * ts_to_ns_),
                                              0., // time precision
                                              time_slice,
                                              digi.getHPTDCErrorFlags(),
                                              digi.getMultipleHit() ) );
    }
  }
}
