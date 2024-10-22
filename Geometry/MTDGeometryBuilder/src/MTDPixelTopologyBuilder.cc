//#define EDM_ML_DEBUG

// Make the change for "big" pixels. 3/06 d.k.
#include "Geometry/MTDGeometryBuilder/interface/MTDPixelTopologyBuilder.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

MTDPixelTopologyBuilder::MTDPixelTopologyBuilder(void) {}

PixelTopology* MTDPixelTopologyBuilder::build(const Bounds* bs,
                                              int pixelROCRows,  // Num of Rows per ROC
                                              int pixelROCCols,  // Num of Cols per ROC
                                              int pixelROCsInX,
                                              int pixelROCsInY,
                                              int GAPxInterpad,
                                              int GAPxBorder,
                                              int GAPyInterpad,
                                              int GAPyBorder) {
  float width = bs->width();    // module width = Xsize
  float length = bs->length();  // module length = Ysize

  // Number of pixel rows (x) and columns (y) per module
  int nrows = pixelROCRows * pixelROCsInX;
  int ncols = pixelROCCols * pixelROCsInY;

  float pitchX = width / nrows;
  float pitchY = length / ncols;

  float micronsTocm = 1e-4;
  float gapxinterpad = float(GAPxInterpad) * micronsTocm;  //Convert to cm
  float gapyinterpad = float(GAPyInterpad) * micronsTocm;  //Convert to cm
  float gapxborder = float(GAPxBorder) * micronsTocm;      //Convert to cm
  float gapyborder = float(GAPyBorder) * micronsTocm;      //Convert to cm

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDPixelTopologyBuilder")
      << std::fixed << "Building topology for module of width(X) = " << std::setw(10) << width
      << " length(Y) = " << std::setw(10) << length << "\n Rows per ROC   = " << std::setw(10) << pixelROCRows
      << " Cols per ROC   = " << std::setw(10) << pixelROCCols << "\n ROCs in X      = " << std::setw(10)
      << pixelROCsInX << " ROCs in Y      = " << std::setw(10) << pixelROCsInY
      << "\n # pixel rows X = " << std::setw(10) << nrows << " # pixel cols Y = " << std::setw(10) << ncols
      << "\n pitch in X     = " << std::setw(10) << pitchX << " # pitch in Y   = " << std::setw(10) << pitchY
      << "\n Interpad gap in X   = " << std::setw(10) << gapxinterpad << " # Interpad gap in Y   = " << std::setw(10)
      << gapyinterpad << "\n Border gap in X   = " << std::setw(10) << gapxborder
      << " # Border gap in Y   = " << std::setw(10) << gapyborder;
#endif

  return (new RectangularMTDTopology(nrows,
                                     ncols,
                                     pitchX,
                                     pitchY,
                                     pixelROCRows,  // (int)rocRow
                                     pixelROCCols,  // (int)rocCol
                                     pixelROCsInX,
                                     pixelROCsInY,
                                     gapxinterpad,
                                     gapxborder,
                                     gapyinterpad,
                                     gapyborder));
}
