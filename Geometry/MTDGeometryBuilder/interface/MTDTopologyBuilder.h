#ifndef Geometry_MTDGeometryBuilder_MTDTopologyBuilder_H
#define Geometry_MTDGeometryBuilder_MTDTopologyBuilder_H

#include <string>
class PixelTopology;
class Bounds;

/**
 * Called by GeomTopologyBuilder, chooses the right topology for Pixels.
 */

class MTDTopologyBuilder {
public:
  MTDTopologyBuilder();

  PixelTopology* build(const Bounds* bounds,
                       int ROWS_PER_ROC,  // Num of Rows per ROC
                       int COLS_PER_ROC,  // Num of Cols per ROC
                       int ROCS_X,
                       int ROCS_Y,
                       int GAPxInterpad,  //This value is given in microns
                       int GAPxBorder,    //This value is given in microns
                       int GAPyInterpad,  //This value is given in microns
                       int GAPyBorder);   //This value is given in microns
};

#endif
