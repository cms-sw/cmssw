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
CTPPSDiamondRecHitProducerAlgorithm::build( const edm::DetSet<CTPPSDiamondDigi>& input, edm::DetSet<CTPPSDiamondRecHit>& output )
{
  for ( edm::DetSet<CTPPSDiamondDigi>::const_iterator it = input.begin(); it != input.end(); ++it )
  {
    const int t = it->getLeadingEdge(),
              t0 = ( t-t_shift_ ) % 1024,
              time_slice = ( t-t_shift_ ) / 1024;
    if ( t==0 ) { continue; }
    const double x_pos = 0.,
                 x_width = 0.,
                 y_pos = 0.,
                 y_width = 0.;
    const CTPPSDiamondRecHit rechit( x_pos, x_width, y_pos, y_width, ( t0 * ts_to_ns_ ), ( it->getTrailingEdge()-t0 ) * ts_to_ns_, time_slice, it->getHPTDCErrorFlags() );
    output.push_back( rechit );
    //output.push_back(TotemRPRecHit(rp_topology_.GetHitPositionInReadoutDirection(it->getCenterStripPosition()), nominal_sigma));
  }
}
