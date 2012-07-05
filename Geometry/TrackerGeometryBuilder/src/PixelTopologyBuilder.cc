// Make the change for "big" pixels. 3/06 d.k.
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

PixelTopologyBuilder::PixelTopologyBuilder( void )
  : thePixelROCRows( 0 ),
    thePixelROCCols( 0 ),
    thePixelROCsInX( 0 ),
    thePixelROCsInY( 0 )
{}

PixelTopology*
PixelTopologyBuilder::build( const Bounds* bs, std::string /* part */,
			     bool upgradeGeometry,
			     int ROWS_PER_ROC, // Num of Rows per ROC
			     int COLS_PER_ROC, // Num of Cols per ROC
			     int BIG_PIX_PER_ROC_X, // in x direction, rows. BIG_PIX_PER_ROC_X = 0 for SLHC
			     int BIG_PIX_PER_ROC_Y, // in y direction, cols. BIG_PIX_PER_ROC_Y = 0 for SLHC
			     int ROCS_X, int ROCS_Y )
{
  thePixelROCRows = ROWS_PER_ROC; // number of pixel rows per ROC
  thePixelROCsInX = ROCS_X;       // number of ROCs per module in x
  thePixelROCCols = COLS_PER_ROC; // number of pixel cols in ROC
  thePixelROCsInY = ROCS_Y;       // number of ROCs per module in y

  float width = bs->width();   // module width = Xsize
  float length = bs->length(); // module length = Ysize

  // Number of pixel rows (x) and columns (y) per module
  int nrows = int(thePixelROCRows * thePixelROCsInX);
  int ncols = int(thePixelROCCols * thePixelROCsInY);

  // Take into account the large edge pixles
  // 1 big pixel per ROC
  float pitchX = width /(float(nrows)+thePixelROCsInX*BIG_PIX_PER_ROC_X); 
  // 2 big pixels per ROC
  float pitchY = length/(float(ncols)+thePixelROCsInY*BIG_PIX_PER_ROC_Y);

  return ( new RectangularPixelTopology( nrows, ncols, pitchX, pitchY,
					 upgradeGeometry,
					 ROWS_PER_ROC, // (int)rocRow
					 COLS_PER_ROC, // (int)rocCol
					 BIG_PIX_PER_ROC_X,
					 BIG_PIX_PER_ROC_Y,
					 ROCS_X, ROCS_Y )); // (int)rocInX, (int)rocInY
}
