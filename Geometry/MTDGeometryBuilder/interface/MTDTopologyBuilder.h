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
		       bool upgradeGeometry,
		       int ROWS_PER_ROC, // Num of Rows per ROC
		       int COLS_PER_ROC, // Num of Cols per ROC
		       int BIG_PIX_PER_ROC_X, // in x direction, rows. BIG_PIX_PER_ROC_X = 0 for SLHC
		       int BIG_PIX_PER_ROC_Y, // in y direction, cols. BIG_PIX_PER_ROC_Y = 0 for SLHC
		       int ROCS_X, int ROCS_Y);
};

#endif
