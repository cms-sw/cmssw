#include "DataFormats/Phase2ITPixelCluster/interface/Phase2ITPixelCluster.h"

//---------------------------------------------------------------------------
//!  \class Phase2ITPixelCluster
//!  \brief Pixel cluster -- collection of pixels with ADC counts
//!
//!  Class to contain and store all the topological information of pixel clusters:
//!  charge, global size, size and the barycenter in x and y
//!  local directions. It builds a vector of SiPixel (which is
//!  an inner class) and a container of channels. 
//!
//!  March 2007: Edge pixel methods moved to RectangularPixelTopology (V.Chiochia)
//!  May   2008: Offset based packing (D.Fehling / A. Rizzi)  
//!  \author Petar Maksimovic, JHU
//---------------------------------------------------------------------------

constexpr unsigned int Phase2ITPixelCluster::MAXSPAN;

Phase2ITPixelCluster::Phase2ITPixelCluster( const Phase2ITPixelCluster::PixelPos& pix, uint32_t adc) :
  thePixelRow(pix.row()),
  thePixelCol(pix.col()),
    // ggiurgiu@fnal.gov, 01/05/12
  // Initialize the split cluster errors to un-physical values.
  // The CPE will check these errors and if they are not un-physical, 
  // it will recognize the clusters as split and assign these (increased) 
  // errors to the corresponding rechit. 
  err_x(-99999.9),
  err_y(-99999.9)
{
  // First pixel in this cluster.
  thePixelADC.push_back( adc );
  thePixelOffset.push_back(0 );
  thePixelOffset.push_back(0 );
}

void Phase2ITPixelCluster::add( const Phase2ITPixelCluster::PixelPos& pix, uint32_t adc) {
  
  uint32_t ominRow = minPixelRow();
  uint32_t ominCol = minPixelCol();
  bool recalculate = false;
  
  uint32_t minRow = ominRow;
  uint32_t minCol = ominCol;
  
  if (pix.row() < minRow) {
    minRow = pix.row();
    recalculate = true;
  }
  if (pix.col() < minCol) {
    minCol = pix.col();
    recalculate = true;
  }
  
  if (recalculate) {
    int maxCol = 0;
    int maxRow = 0;
    int isize = thePixelADC.size();
    for (int i=0; i<isize; ++i) {
      int xoffset = thePixelOffset[i*2]  + ominRow - minRow;
      int yoffset = thePixelOffset[i*2+1]  + ominCol -minCol;
      thePixelOffset[i*2] = std::min(int(MAXSPAN),xoffset);
      thePixelOffset[i*2+1] = std::min(int(MAXSPAN),yoffset);
      if (xoffset > maxRow) maxRow = xoffset; 
      if (yoffset > maxCol) maxCol = yoffset; 
    }
    packRow(minRow,maxRow);
    packCol(minCol,maxCol);
  }
  
  if ( (!overflowRow()) && pix.row() > maxPixelRow()) 
    packRow(minRow,pix.row()-minRow);
  
  if ( (!overflowCol()) && pix.col() > maxPixelCol())
    packCol(minCol,pix.col()-minCol);
  
  thePixelADC.push_back( adc );
  thePixelOffset.push_back( std::min(MAXSPAN,pix.row() - minRow) );
  thePixelOffset.push_back( std::min(MAXSPAN,pix.col() - minCol) );
}
