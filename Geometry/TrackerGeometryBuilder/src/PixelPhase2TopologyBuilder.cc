// Make the change for "big" pixels. 3/06 d.k.
#include "Geometry/TrackerGeometryBuilder/interface/PixelPhase2TopologyBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelPhase2Topology.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

PixelPhase2TopologyBuilder::PixelPhase2TopologyBuilder(void) {}

PixelTopology* PixelPhase2TopologyBuilder::build(const Bounds* bs,
                                                 int pixelROCRows,       // Num of Rows per ROC
                                                 int pixelROCCols,       // Num of Cols per ROC
                                                 int BIG_PIX_PER_ROC_X,  // in x direction, rows.
                                                 int BIG_PIX_PER_ROC_Y,  // in y direction, cols.
                                                 float BIG_PIX_PITCH_X,
                                                 float BIG_PIX_PITCH_Y,
                                                 int pixelROCsInX,
                                                 int pixelROCsInY) {
  float width = bs->width();    // module width = Xsize
  float length = bs->length();  // module length = Ysize

  // Number of pixel rows (x) and columns (y) per module
  int nrows = pixelROCRows * pixelROCsInX;
  int ncols = pixelROCCols * pixelROCsInY;

  // Take into account the large edge pixels
  float pitchX =
      (width - pixelROCsInX * BIG_PIX_PER_ROC_X * BIG_PIX_PITCH_X) / (nrows - pixelROCsInX * BIG_PIX_PER_ROC_X);
  float pitchY =
      (length - pixelROCsInY * BIG_PIX_PER_ROC_Y * BIG_PIX_PITCH_Y) / (ncols - pixelROCsInY * BIG_PIX_PER_ROC_Y);
  if (BIG_PIX_PER_ROC_X == 0)
    BIG_PIX_PITCH_X =
        pitchX;  // should then be either the exact one for Big Pixels or the expected one in the old geometry
  if (BIG_PIX_PER_ROC_Y == 0)
    BIG_PIX_PITCH_Y = pitchY;

  return (new RectangularPixelPhase2Topology(nrows,
                                             ncols,
                                             pitchX,
                                             pitchY,
                                             pixelROCRows,  // (int)rocRow
                                             pixelROCCols,  // (int)rocCol
                                             BIG_PIX_PER_ROC_X,
                                             BIG_PIX_PER_ROC_Y,
                                             BIG_PIX_PITCH_X,
                                             BIG_PIX_PITCH_Y,
                                             pixelROCsInX,
                                             pixelROCsInY));  // (int)rocInX, (int)rocInY
}
