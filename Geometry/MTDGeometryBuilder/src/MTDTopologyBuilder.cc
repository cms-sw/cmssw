//#define EDM_ML_DEBUG

// Make the change for "big" pixels. 3/06 d.k.
#include "Geometry/MTDGeometryBuilder/interface/MTDTopologyBuilder.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

MTDTopologyBuilder::MTDTopologyBuilder(void) {}

PixelTopology* MTDTopologyBuilder::build(const Bounds* bs,
                                         bool upgradeGeometry,
                                         int pixelROCRows,       // Num of Rows per ROC
                                         int pixelROCCols,       // Num of Cols per ROC
                                         int BIG_PIX_PER_ROC_X,  // in x direction, rows. BIG_PIX_PER_ROC_X = 0 for SLHC
                                         int BIG_PIX_PER_ROC_Y,  // in y direction, cols. BIG_PIX_PER_ROC_Y = 0 for SLHC
                                         int pixelROCsInX,
                                         int pixelROCsInY) {
  float width = bs->width();    // module width = Xsize
  float length = bs->length();  // module length = Ysize

  // Number of pixel rows (x) and columns (y) per module
  int nrows = pixelROCRows * pixelROCsInX;
  int ncols = pixelROCCols * pixelROCsInY;

  // Take into account the large edge pixles
  // 1 big pixel per ROC
  float pitchX = width / (nrows + pixelROCsInX * BIG_PIX_PER_ROC_X);
  // 2 big pixels per ROC
  float pitchY = length / (ncols + pixelROCsInY * BIG_PIX_PER_ROC_Y);

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDTopologyBuilder") << std::fixed << "Building topology for module of width(X) = " << std::setw(10)
                                     << width << " length(Y) = " << std::setw(10) << length
                                     << "\n Rows per ROC   = " << std::setw(10) << pixelROCRows
                                     << " Cols per ROC   = " << std::setw(10) << pixelROCCols
                                     << "\n ROCs in X      = " << std::setw(10) << pixelROCsInX
                                     << " ROCs in Y      = " << std::setw(10) << pixelROCsInY
                                     << "\n # pixel rows X = " << std::setw(10) << nrows
                                     << " # pixel cols Y = " << std::setw(10) << ncols
                                     << "\n pitch in X     = " << std::setw(10) << pitchX
                                     << " # pitch in Y   = " << std::setw(10) << pitchY;
#endif

  return (new RectangularMTDTopology(nrows,
                                     ncols,
                                     pitchX,
                                     pitchY,
                                     upgradeGeometry,
                                     pixelROCRows,  // (int)rocRow
                                     pixelROCCols,  // (int)rocCol
                                     BIG_PIX_PER_ROC_X,
                                     BIG_PIX_PER_ROC_Y,
                                     pixelROCsInX,
                                     pixelROCsInY));  // (int)rocInX, (int)rocInY
}
