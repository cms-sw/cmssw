// Make the change for "big" pixels. 3/06 d.k.
#include <iostream>

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
PixelTopologyBuilder::build( const Bounds* bs, double rocRow, double rocCol, double rocInX, double rocInY, std::string /* part */,
			     bool upgradeGeometry,
			     int ROWS_PER_ROC, // Num of Rows per ROC
			     int COLS_PER_ROC, // Num of Cols per ROC
			     int BIG_PIX_PER_ROC_X, // in x direction, rows. BIG_PIX_PER_ROC_X = 0 for SLHC
			     int BIG_PIX_PER_ROC_Y, // in y direction, cols. BIG_PIX_PER_ROC_Y = 0 for SLHC
			     int ROCS_X, int ROCS_Y )
{
  thePixelROCRows = rocRow; // number of pixel rows per ROC
  thePixelROCsInX = rocInX; // number of ROCs per module in x
  thePixelROCCols = rocCol; // number of pixel cols in ROC
  thePixelROCsInY = rocInY; // number of ROCs per module in y

  float width = bs->width(); // module width = Xsize
  float length = bs->length(); // module length = Ysize

  // Number of pixel rows (x) and columns (y) per module
  int nrows = int(thePixelROCRows * thePixelROCsInX);
  int ncols = int(thePixelROCCols * thePixelROCsInY);

  // For all pixels having same size (old topology)
  //float pitchX = width/float(nrows); 
  //float pitchY = length/float(ncols);

//   // temporary before we find a better way to do this 
//   const int BIG_PIX_PER_ROC_X = 1; // 1 big pixel  in x direction, rows
//   const int BIG_PIX_PER_ROC_Y = 2; // 2 big pixels in y direction, cols

  // Take into account the large edge pixles
  // 1 big pixel per ROC
  float pitchX = width /(float(nrows)+thePixelROCsInX*BIG_PIX_PER_ROC_X); 
  // 2 big pixels per ROC
  float pitchY = length/(float(ncols)+thePixelROCsInY*BIG_PIX_PER_ROC_Y);

  //std::cout<<"Build Pixel Topology: row/cols = "<<nrows<<"/"<<ncols
  //   <<" sizeX/Y = "<<width<<"/"<<length
  //   <<" pitchX/Y = "<<pitchX<<"/"<<pitchY
  //   <<" ROCsX/Y = "<<thePixelROCsInX<<"/"<<thePixelROCsInY
  //   <<" per ROC row/cols = "<<thePixelROCRows<<"/"<<thePixelROCCols
  //   <<" big pixels "<<BIG_PIX_PER_ROC_X<<"/"<<BIG_PIX_PER_ROC_Y
  //   <<std::endl;   

  return ( upgradeGeometry
	   ? ( new RectangularPixelTopology( nrows, ncols, pitchX, pitchY, (int)rocInX, (int)rocInY, (int)rocRow, (int)rocCol,
					     upgradeGeometry,
					     ROWS_PER_ROC,
					     COLS_PER_ROC,
					     BIG_PIX_PER_ROC_X,
					     BIG_PIX_PER_ROC_Y,
					     ROCS_X, ROCS_Y ))
	   : ( new RectangularPixelTopology( nrows, ncols, pitchX, pitchY,
					     upgradeGeometry,
					     ROWS_PER_ROC,
					     COLS_PER_ROC,
					     BIG_PIX_PER_ROC_X,
					     BIG_PIX_PER_ROC_Y,
					     ROCS_X, ROCS_Y )));
}


