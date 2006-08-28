#ifndef PixelROC_H
#define PixelROC_H

#include <boost/cstdint.hpp>
#include <string>

/** \class PixelROC
 * Represents ReadOut Chip of DetUnit. 
 * Converts pixel coordinates from Local (in ROC) to Global (in DetUnit).
 * The Local coordinates are double column (dcol) and pixel index in dcol.
 * The Global coordinates are row and column in DetUnit.
 */

class PixelROC {
public:

  /// dummy
  PixelROC(){}

  /// ctor with offsets in DU (units of ROC)
  PixelROC( 
      uint32_t du, int idDU, int idLk, 
      int rocInX, int rocInY); 


  /// id of this ROC in DetUnit (representing pixel module) according 
  /// to PixelDatabase. 
  int idInDetUnit() const { return theIdDU; }

  /// id of this ROC in parent Link.
  int idInLink() const { return theIdLk; }

  /// return the DetUnit to which this ROC belongs to.
  uint32_t rawId() const { return theDetUnit; }

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
    int icol, irow;
    if (loc.pxid < theNRows) {
      icol = 0;
      irow = loc.pxid;
    }
    else {
      icol = 1;
      irow = 2*theNRows - loc.pxid-1;
    }
    GlobalPixel res;
    res.row = theNRows*theRocInY + irow;
    res.col = theNCols*theRocInX + loc.dcol * 2 + icol;
    return res;
  }

  /// check if position is inside this ROC
  bool inside(const LocalPixel & lp) const;

  /// check if position inside this ROC
  bool inside(const GlobalPixel & gp) const { return inside(toLocal(gp)); }

  std::string print(int depth = 0) const;


  /// number of rows in ROC
  static int rows() { return theNRows; }
  /// number of columns in ROC 
  static int cols() { return theNCols; }

private:
  uint32_t theDetUnit;
  int theIdDU, theIdLk;
  int theRocInX, theRocInY; // offsets in DU (in units of ROC);
  static int theNRows, theNCols; 
};

#endif
