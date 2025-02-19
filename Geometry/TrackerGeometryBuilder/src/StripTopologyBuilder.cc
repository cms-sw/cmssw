#include "Geometry/TrackerGeometryBuilder/interface/StripTopologyBuilder.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/RadialStripTopology.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

StripTopologyBuilder::StripTopologyBuilder( void )
  : theAPVNumb( 0.0 )
{}

StripTopology*
StripTopologyBuilder::build( const Bounds* bs, double apvnumb, std::string part )
{
  theAPVNumb = apvnumb;

  StripTopology* result = 0;
  if( part == "barrel" )
  {
    result = constructBarrel( bs->length(), bs->width());
  }
  else
  {
    const TrapezoidalPlaneBounds* topo = dynamic_cast<const TrapezoidalPlaneBounds*>( bs );
    if( topo )
    {      
      int yAx = topo->yAxisOrientation();
      result = constructForward( bs->length(), bs->width(), bs->widthAtHalfLength(), yAx );
    }
  }
  return result;
}

StripTopology*
StripTopologyBuilder::constructBarrel( float length, float width )
{
  int nstrip = int( 128 * theAPVNumb );
  float pitch = width / nstrip;
  
  return new RectangularStripTopology( nstrip, pitch, length );
}

StripTopology*
StripTopologyBuilder::constructForward( float length, float width, float widthAtHalf, int yAxOr )
{
  int nstrip = int( 128 * theAPVNumb );
  float rCross = widthAtHalf * length / ( 2 * ( width - widthAtHalf ));
  float aw = atan2( widthAtHalf / 2., static_cast<double>( rCross )) / ( nstrip / 2 );
  return new RadialStripTopology( nstrip, aw, length, rCross, yAxOr );
}

