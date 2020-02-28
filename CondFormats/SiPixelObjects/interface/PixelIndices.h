#ifndef TP_PIXELINDICES_H
#define TP_PIXELINDICES_H

#include <iostream>

/**
 * Numbering of the pixels inside the readout chip (ROC).
 * There is a column index and a row index.
 * In the barrel the row index is in the global rfi direction (local X) and
 * the column index is in the global z (local Y) direction.
 * In the endcaps the row index is in the global r direction (local X) and
 * the column index in the gloabl rfi (local Y) direction.
 * The defaults are specific to 100*150 micron pixels.
 * 
 * Some methods are declared as static and can be used without class 
 * instantiation. Others need the construstor to setup the Module size
 * parameters. These parameters are only used for error checking.
 * d.k. 10/2005
 */

namespace {
  // A few constants just for error checking
  // The maximum number of ROCs in the X (row) direction per sensor.
  const int maxROCsInX = 2;  //
  // The maximum number of ROCs in the Y (column) direction per sensor.
  const int maxROCsInY = 8;  //
  // The nominal number of double columns per ROC is 26.
  const int DColsPerROC = 26;
  // Default ROC size
  const int ROCSizeInX = 80;  // ROC row size in pixels
  const int ROCSizeInY = 52;  // ROC col size in pixels
  // Default DET barrel size
  const int defaultDetSizeInX = 160;  // Det barrel row size in pixels
  const int defaultDetSizeInY = 416;  // Det barrel col size in pixels

  // Check the limits
  const bool TP_CHECK_LIMITS = true;
}  // namespace

class PixelIndices {
public:
  //*********************************************************************
  // Constructor with the ROC size fixed to the default.
  PixelIndices(const int colsInDet, const int rowsInDet) : theColsInDet(colsInDet), theRowsInDet(rowsInDet) {
    theChipsInX = theRowsInDet / ROCSizeInX;  // number of ROCs in X
    theChipsInY = theColsInDet / ROCSizeInY;  // number of ROCs in Y

    if (TP_CHECK_LIMITS) {
      if (theChipsInX < 1 || theChipsInX > maxROCsInX)
        std::cout << " PixelIndices: Error in ROCsInX " << theChipsInX << " " << theRowsInDet << " " << ROCSizeInX
                  << std::endl;
      if (theChipsInY < 1 || theChipsInY > maxROCsInY)
        std::cout << " PixelIndices: Error in ROCsInY " << theChipsInY << " " << theColsInDet << " " << ROCSizeInY
                  << std::endl;
    }
  }
  //************************************************************************
  ~PixelIndices() {}
  //***********************************************************************

  inline int numberOfROCsInX(void) { return theChipsInX; }
  inline int numberOfROCsInY(void) { return theChipsInY; }

  //***********************************************************************

  void print(void) const {
    std::cout << " Pixel det with " << theChipsInX << " chips in x and " << theChipsInY << " in y " << std::endl;
    std::cout << " Pixel rows " << theRowsInDet << " and columns " << theColsInDet << std::endl;
    std::cout << " Rows in one chip " << ROCSizeInX << " and columns " << ROCSizeInY << std::endl;
    std::cout << " Double columns per ROC " << DColsPerROC << std::endl;
  }

  //********************************************************************
  // Convert dcol & pix indices to ROC col and row
  // Decoding from "Weber" pixel addresses to rows for PSI46
  // dcol = 0 - 25
  // pix = 2 - 161, zigzag pattern.
  // colAdd = 0-51   ! col&row start from 0
  // rowAdd = 0-79
  inline static int convertDcolToCol(const int dcol, const int pix, int& colROC, int& rowROC) {
    if (TP_CHECK_LIMITS) {
      if (dcol < 0 || dcol >= DColsPerROC || pix < 2 || pix > 161) {
        std::cout << "PixelIndices: wrong dcol or pix " << dcol << " " << pix << std::endl;
        rowROC = -1;  // dummy row Address
        colROC = -1;  // dummy col Address
        return -1;    // Signal error
      }
    }

    // First find if we are in the first or 2nd col of a dcol.
    int colEvenOdd = pix % 2;  // module(2), 0-1st sol, 1-2nd col.
    // Transform
    colROC = dcol * 2 + colEvenOdd;   // col address, starts from 0
    rowROC = abs(int(pix / 2) - 80);  // row addres, starts from 0

    if (TP_CHECK_LIMITS) {
      if (colROC < 0 || colROC >= ROCSizeInY || rowROC < 0 || rowROC >= ROCSizeInX) {
        std::cout << "PixelIndices: wrong col or row " << colROC << " " << rowROC << " " << dcol << " " << pix
                  << std::endl;
        rowROC = -1;  // dummy row Address
        colROC = -1;  // dummy col Address
        return -1;
      }
    }
    return 0;
  }

  //********************************************************************
  // colROC, rowROC are coordinates in the ROC frame, for ROC=rocId
  // (Start from 0).
  // cols, row are coordinates in the module frame, start from 0.
  // row is X, col is Y.
  // At the moment this works only for modules read with a single TBM.
  int transformToModule(const int colROC, const int rowROC, const int rocId, int& col, int& row) const {
    if (TP_CHECK_LIMITS) {
      if (colROC < 0 || colROC >= ROCSizeInY || rowROC < 0 || rowROC >= ROCSizeInX) {
        std::cout << "PixelIndices: wrong index " << colROC << " " << rowROC << std::endl;
        return -1;
      }
    }

    // The transformation depends on the ROC-ID
    if (rocId >= 0 && rocId < 8) {
      row = 159 - rowROC;
      //col = rocId*52 + colROC;
      col = (8 - rocId) * ROCSizeInY - colROC - 1;
    } else if (rocId >= 8 && rocId < 16) {
      row = rowROC;
      //col = (16-rocId)*52 - colROC - 1;
      col = (rocId - 8) * ROCSizeInY + colROC;
    } else {
      std::cout << "PixelIndices: wrong ROC ID " << rocId << std::endl;
      return -1;
    }
    if (TP_CHECK_LIMITS) {
      if (col < 0 || col >= (ROCSizeInY * theChipsInY) || row < 0 || row >= (ROCSizeInX * theChipsInX)) {
        std::cout << "PixelIndices: wrong index " << col << " " << row << std::endl;
        return -1;
      }
    }

    return 0;
  }
  //**************************************************************************
  // Transform from the module indixes to the ROC indices.
  // col, row - indices in the Module
  // rocId - roc index
  // colROC, rowROC - indices in the ROC frame.
  int transformToROC(const int col, const int row, int& rocId, int& colROC, int& rowROC) const {
    if (TP_CHECK_LIMITS) {
      if (col < 0 || col >= (ROCSizeInY * theChipsInY) || row < 0 || row >= (ROCSizeInX * theChipsInX)) {
        std::cout << "PixelIndices: wrong index 3 " << std::endl;
        return -1;
      }
    }

    // Get the 2d ROC coordinate
    int chipX = row / ROCSizeInX;  // row index of the chip 0-1
    int chipY = col / ROCSizeInY;  // col index of the chip 0-7

    // Get the ROC id from the 2D index
    rocId = rocIndex(chipX, chipY);
    if (TP_CHECK_LIMITS && (rocId < 0 || rocId >= 16)) {
      std::cout << "PixelIndices: wrong roc index " << rocId << std::endl;
      return -1;
    }
    // get the local ROC coordinates
    rowROC = (row % ROCSizeInX);  // row in chip
    colROC = (col % ROCSizeInY);  // col in chip

    if (rocId < 8) {  // For lower 8 ROCs the coordinates are reversed
      colROC = 51 - colROC;
      rowROC = 79 - rowROC;
    }

    if (TP_CHECK_LIMITS) {
      if (colROC < 0 || colROC >= ROCSizeInY || rowROC < 0 || rowROC >= ROCSizeInX) {
        std::cout << "PixelIndices: wrong index " << colROC << " " << rowROC << std::endl;
        return -1;
      }
    }

    return 0;
  }
  //***********************************************************************
  // Calculate a single number ROC index from the 2 ROC indices (coordinates)
  // chipX and chipY.
  // Goes from 0 to 15.
  inline static int rocIndex(const int chipX, const int chipY) {
    int rocId = -1;
    if (TP_CHECK_LIMITS) {
      if (chipX < 0 || chipX >= 2 || chipY < 0 || chipY >= 8) {
        std::cout << "PixelChipIndices: wrong index " << chipX << " " << chipY << std::endl;
        return -1;
      }
    }
    if (chipX == 0)
      rocId = chipY + 8;  // should be 8-15
    else if (chipX == 1)
      rocId = 7 - chipY;  // should be 0-7

    if (TP_CHECK_LIMITS) {
      if (rocId < 0 || rocId >= (maxROCsInX * maxROCsInY)) {
        std::cout << "PixelIndices: Error in ROC index " << rocId << std::endl;
        return -1;
      }
    }
    return rocId;
  }
  //**************************************************************************
  // Calculate the dcol in ROC from the col in ROC frame.
  // dcols go from 0 to 25.
  inline static int DColumn(const int colROC) {
    int dColumnId = (colROC) / 2;  // double column 0-25
    if (TP_CHECK_LIMITS) {
      if (dColumnId < 0 || dColumnId >= 26) {
        std::cout << "PixelIndices: wrong dcol index  " << dColumnId << " " << colROC << std::endl;
        return -1;
      }
    }
    return dColumnId;
  }
  //*************************************************************************
  // Calcuulate the global dcol index within a module
  // Usefull only forin efficiency calculations.
  inline static int DColumnInModule(const int dcol, const int chipIndex) {
    int dcolInMod = dcol + chipIndex * 26;
    return dcolInMod;
  }

  // This is routines to generate ROC channel number
  // Only limited use to store ROC pixel indices for calibration
  inline static int pixelToChannelROC(const int rowROC, const int colROC) {
    return (rowROC << 6) | colROC;  // reserve 6 bit for col ROC index 0-52
  }
  inline static std::pair<int, int> channelToPixelROC(const int chan) {
    int rowROC = (chan >> 6) & 0x7F;  // reserve 7 bits for row ROC index 0-79
    int colROC = chan & 0x3F;
    return std::pair<int, int>(rowROC, colROC);
  }

  //***********************************************************************
private:
  int theColsInDet;  // Columns per Det
  int theRowsInDet;  // Rows per Det
  int theChipsInX;   // Chips in det in X (column direction)
  int theChipsInY;   // Chips in det in Y (row direction)
};

#endif
