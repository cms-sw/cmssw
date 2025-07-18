#ifndef Geometry_TrackerGeometryBuilder_PixelPhase2TopologyBuilder_H
#define Geometry_TrackerGeometryBuilder_PixelPhase2TopologyBuilder_H

#include <string>
class PixelTopology;
class Bounds;

/**
 * Called by GeomTopologyBuilder, chooses the right topology for Pixels.
 */

class PixelPhase2TopologyBuilder {
public:
  PixelPhase2TopologyBuilder();

  PixelTopology* build(const Bounds* bounds,
                       int ROWS_PER_ROC,       // Num of Rows per ROC
                       int COLS_PER_ROC,       // Num of Cols per ROC
                       int BIG_PIX_PER_ROC_X,  // in x direction, rows
                       int BIG_PIX_PER_ROC_Y,  // in y direction, cols
                       float BIG_PIX_PITCH_X,
                       float BIG_PIX_PITCH_Y,
                       int ROCS_X,
                       int ROCS_Y);
};

#endif
