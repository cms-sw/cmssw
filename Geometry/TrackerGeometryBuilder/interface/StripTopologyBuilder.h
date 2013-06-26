#ifndef Geometry_TrackerGeometryBuilder_StripTopologyBuilder_H
#define Geometry_TrackerGeometryBuilder_StripTopologyBuilder_H

#include <string>

class StripTopology;
class Bounds;

/**
 * Called by GeomTopologyBuilder, chooses the right topology for Strips.
 */

class StripTopologyBuilder
{
public:

  StripTopologyBuilder( void );

  StripTopology* build( const Bounds*, double, std::string );

private:

  double theAPVNumb;

  StripTopology* constructBarrel( float , float );
  StripTopology* constructForward( float, float, float, int );
};

#endif
