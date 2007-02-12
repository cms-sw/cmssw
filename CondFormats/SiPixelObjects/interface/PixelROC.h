#ifndef SiPixelObjects_PixelROC_H
#define SiPixelObjects_PixelROC_H

#include "CondFormats/SiPixelObjects/interface/FrameConversion.h"
#include <boost/cstdint.hpp>
#include <string>

/** \class PixelROC
 * Represents ReadOut Chip of DetUnit. 
 * Converts pixel coordinates from Local (in ROC) to Global (in DetUnit).
 * The Local coordinates are double column (dcol) and pixel index in dcol.
 * The Global coordinates are row and column in DetUnit.
 */


namespace sipixelobjects {

class PixelROC {
public:

  /// dummy
  PixelROC() : theDetUnit(0), theIdDU(0), theIdLk(0), 
               theRowOffset(0),theRowSlopeSign(0), theColOffset(0), theColSlopeSign(0) 
  { }

  /// ctor with DetUnit id, 
  /// ROC number in DU (given by token passage), 
  /// ROC number in Link (given by token passage),
  /// conversion of this  ROC do DetUnit 
  PixelROC( uint32_t du, int idInDU, int idLk, const FrameConversion & frame);

  /// return the DetUnit to which this ROC belongs to.
  uint32_t rawId() const { return theDetUnit; }

  /// id of this ROC in DetUnit etermined by token path 
  int idInDetUnit() const { return theIdDU; }

  /// id of this ROC in parent Link.
  int idInLink() const { return theIdLk; }

  /// local coordinates in this ROC (double column, pixelid in double column) 
  struct LocalPixel { int dcol, pxid; };
  /// global coordinates (row and column in DetUnit, as in PixelDigi)
  struct GlobalPixel { int row; int col; };

  /// converts DU position to local. 
  /// If GlobalPixel is outside ROC the resulting LocalPixel is not inside ROC.
  /// (call to inside(..) recommended)
  LocalPixel  toLocal(const GlobalPixel & gp) const;

  /// converts LocalPixel in ROC to DU coordinates. 
  /// LocalPixel must be inside ROC. Otherwise result is meaningless
  GlobalPixel toGlobal(const LocalPixel & loc) const {
    int rocCol = loc.dcol*2 + loc.pxid%2;
    int rocRow = theNRows-loc.pxid/2;
    GlobalPixel result;
    FrameConversion conv(theRowOffset,theRowSlopeSign,theColOffset,theColSlopeSign);
    result.col    = conv.collumn().convert(rocCol);
    result.row    = conv.row().convert(rocRow);
    return result;
  }

  /// check if position is inside this ROC
  bool inside(const LocalPixel & lp) const;

  /// check if position inside this ROC
  bool inside(const GlobalPixel & gp) const { return inside(toLocal(gp)); }

  /// printout for debug
  std::string print(int depth = 0) const;

  /// number of rows in ROC
  static int rows() { return theNRows; }
  /// number of columns in ROC 
  static int cols() { return theNCols; }

private:
  uint32_t theDetUnit;
  int theIdDU, theIdLk;
  int theRowOffset, theRowSlopeSign; 
  int theColOffset, theColSlopeSign; 

  static int theNRows, theNCols; 
};

}

#endif
