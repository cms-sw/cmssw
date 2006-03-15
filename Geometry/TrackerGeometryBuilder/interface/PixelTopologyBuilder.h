#ifndef Geometry_TrackerGeometryBuilder_PixelTopologyBuilder_H
#define Geometry_TrackerGeometryBuilder_PixelTopologyBuilder_H

#include <string>
class PixelTopology;
class Bounds;

/**
 * Called by GeomTopologyBuilder, chooses the right topology for Pixels.
 */

class PixelTopologyBuilder {
public:

  PixelTopologyBuilder();

  PixelTopology* build(const Bounds* , double ,double ,double ,double ,std::string );

private:

  double thePixelROCRows;
  double thePixelROCCols;
  double thePixelBarrelROCsInX;
  double thePixelBarrelROCsInY;
  
};

#endif
