#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

//---------------------------------------------------------------------------
//!  \class SiPixelCluster
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

SiPixelCluster::SiPixelCluster(const SiPixelCluster::PixelPos& pix, int adc)
    : theMinPixelRow(pix.row()), theMinPixelCol(pix.col()) {
  // First pixel in this cluster.
  thePixelADC.push_back(adc);
  thePixelOffset.push_back(0);
  thePixelOffset.push_back(0);
}

void SiPixelCluster::add(const SiPixelCluster::PixelPos& pix, int adc) {
  int ominRow = minPixelRow();
  int ominCol = minPixelCol();
  bool recalculate = false;

  int minRow = ominRow;
  int minCol = ominCol;

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
    for (int i = 0; i < isize; ++i) {
      int xoffset = thePixelOffset[i * 2] + ominRow - minRow;
      int yoffset = thePixelOffset[i * 2 + 1] + ominCol - minCol;
      thePixelOffset[i * 2] = std::min(int(MAXSPAN), xoffset);
      thePixelOffset[i * 2 + 1] = std::min(int(MAXSPAN), yoffset);
      if (xoffset > maxRow)
        maxRow = xoffset;
      if (yoffset > maxCol)
        maxCol = yoffset;
    }
    packRow(minRow, maxRow);
    packCol(minCol, maxCol);
  }

  if ((!overflowRow()) && pix.row() > maxPixelRow())
    packRow(minRow, pix.row() - minRow);

  if ((!overflowCol()) && pix.col() > maxPixelCol())
    packCol(minCol, pix.col() - minCol);

  thePixelADC.push_back(adc);
  thePixelOffset.push_back(std::min(int(MAXSPAN), pix.row() - minRow));
  thePixelOffset.push_back(std::min(int(MAXSPAN), pix.col() - minCol));
}
