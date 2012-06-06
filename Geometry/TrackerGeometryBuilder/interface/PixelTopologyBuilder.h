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
  double thePixelROCsInX;
  double thePixelROCsInY;
  int m_BIG_PIX_PER_ROC_X;
  int m_BIG_PIX_PER_ROC_Y;
  bool m_upgradeGeometry;
};

#endif
