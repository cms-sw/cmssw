#ifndef Geometry_TrackerGeometryBuilder_GeomTopologyBuilder_H
#define Geometry_TrackerGeometryBuilder_GeomTopologyBuilder_H

#include <string>

class PixelTopology;
class StripTopology;
class Bounds;

/**
 */

class GeomTopologyBuilder {
public:

  GeomTopologyBuilder();

  PixelTopology* buildPixel( const Bounds* , double , double , double , double ,std::string);
  StripTopology* buildStrip( const Bounds* , double ,std::string);

private:

};

#endif
