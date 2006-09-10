#ifndef SiPixelObjects_PixelROC_H
#define SiPixelObjects_PixelROC_H

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
  PixelROC(){}

  /// ctor with offsets in DU (units of ROC)
  PixelROC( 
      uint32_t du, int idDU, int idLk, 
      int rocInX, int rocInY); 

  /// id of this ROC in DetUnit (representing pixel module) according 
  /// to PixelDatabase. 
  int idInDetUnit() const { return theIdDU; }

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
  GlobalPixel toGlobal(const LocalPixel & loc) const;

  /// check if position is inside this ROC
  bool inside(const LocalPixel & lp) const;

  /// check if position inside this ROC
  bool inside(const GlobalPixel & gp) const { return inside(toLocal(gp)); }

  bool inside(int dcol, int pxid) const;
  /// id of this ROC in parent Link.
  int idInLink() const { return theIdLk; }

  /// x position of this ROC in DetUnit (in units of ROCs)
  int x() const { return theRocInX; }

  /// y position of this ROC in DetUnit (in units of ROCs)
  int y() const { return theRocInY; }



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

  int theRowOffset, theRowSlopeSign; 
  int theColOffset, theColSlopeSign; 

};

}

#endif
