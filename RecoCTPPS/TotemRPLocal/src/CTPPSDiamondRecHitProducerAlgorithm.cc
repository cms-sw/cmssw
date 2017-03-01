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
  ts_to_ns_( iConfig.getParameter<double>( "timeSliceNs" ) ),
  t_shift_( iConfig.getParameter<int>( "timeShift" ) )
{}

void
CTPPSDiamondRecHitProducerAlgorithm::build( const TotemRPGeometry* geom, const edm::DetSetVector<CTPPSDiamondDigi>& input, edm::DetSetVector<CTPPSDiamondRecHit>& output )
{
  for ( edm::DetSetVector<CTPPSDiamondDigi>::const_iterator vec = input.begin(); vec != input.end(); ++vec )
  {
    const CTPPSDiamondDetId detid( vec->detId() );

    const DetGeomDesc* det = geom->GetDetector( detid );
    const float x_pos = det->translation().x(),
                x_width = 2.0 * det->params().at( 0 ), // parameters stand for half the size
                y_pos = det->translation().y(),
                y_width = 2.0 * det->params().at( 1 );

    edm::DetSet<CTPPSDiamondRecHit>& rec_hits = output.find_or_insert( detid );

    for ( edm::DetSet<CTPPSDiamondDigi>::const_iterator digi = vec->begin(); digi != vec->end(); ++digi )
    {
      const int t = digi->getLeadingEdge();
      if ( t==0 ) { continue; }

      const int t0 = ( t-t_shift_ ) % 1024,
                time_slice = ( t-t_shift_ ) / 1024;

      rec_hits.push_back( CTPPSDiamondRecHit( x_pos, x_width, y_pos, y_width, // spatial information
                                              ( t0 * ts_to_ns_ ),
                                              ( digi->getTrailingEdge()-t0 ) * ts_to_ns_,
                                              time_slice,
                                              digi->getHPTDCErrorFlags() ) );
    }
  }
}
