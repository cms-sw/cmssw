// Make the change for "big" pixels. 3/06 d.k.
#include "Geometry/MTDGeometryBuilder/interface/MTDTopologyBuilder.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

MTDTopologyBuilder::MTDTopologyBuilder( void )
{}

PixelTopology*
MTDTopologyBuilder::build( const Bounds* bs, 
			   bool upgradeGeometry,
			   int pixelROCRows, // Num of Rows per ROC
			   int pixelROCCols, // Num of Cols per ROC
			   int BIG_PIX_PER_ROC_X, // in x direction, rows. BIG_PIX_PER_ROC_X = 0 for SLHC
			   int BIG_PIX_PER_ROC_Y, // in y direction, cols. BIG_PIX_PER_ROC_Y = 0 for SLHC
			   int pixelROCsInX, int pixelROCsInY )
{
  float width = bs->width();   // module width = Xsize
  float length = bs->length(); // module length = Ysize

  // Number of pixel rows (x) and columns (y) per module
  int nrows = pixelROCRows * pixelROCsInX;
  int ncols = pixelROCCols * pixelROCsInY;

  // Take into account the large edge pixles
  // 1 big pixel per ROC
  float pitchX = width /(nrows+pixelROCsInX*BIG_PIX_PER_ROC_X); 
  // 2 big pixels per ROC
  float pitchY = length/(ncols+pixelROCsInY*BIG_PIX_PER_ROC_Y);
  
  return ( new RectangularMTDTopology( nrows, ncols, pitchX, pitchY,
				       upgradeGeometry,
				       pixelROCRows, // (int)rocRow
				       pixelROCCols, // (int)rocCol
				       BIG_PIX_PER_ROC_X,
				       BIG_PIX_PER_ROC_Y,
				       pixelROCsInX, pixelROCsInY )); // (int)rocInX, (int)rocInY
}
