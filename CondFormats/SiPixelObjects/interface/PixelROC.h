#ifndef SiPixelObjects_PixelROC_H
#define SiPixelObjects_PixelROC_H


#include "CondFormats/SiPixelObjects/interface/FrameConversion.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
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
  PixelROC() : theDetUnit(0), theIdDU(0), theIdLk(0) {} 


  /// ctor with DetUnit id, 
  /// ROC number in DU (given by token passage), 
  /// ROC number in Link (given by token passage),
  PixelROC( uint32_t du, int idInDU, int idLk);

  /// return the DetUnit to which this ROC belongs to.
  uint32_t rawId() const { return theDetUnit; }

  /// id of this ROC in DetUnit etermined by token path 
  unsigned int idInDetUnit() const { return theIdDU; }

  /// id of this ROC in parent Link.
  unsigned int idInLink() const { return theIdLk; }

  /// converts DU position to local. 
  /// If GlobalPixel is outside ROC the resulting LocalPixel is not inside ROC.
  /// (call to inside(..) recommended)
  LocalPixel  toLocal(const GlobalPixel & glo) const {
    int rocRow = theFrameConverter.row().inverse(glo.row);
    int rocCol = theFrameConverter.collumn().inverse(glo.col);
    
    LocalPixel::RocRowCol rocRowCol = {rocRow, rocCol};
    return LocalPixel(rocRowCol);

  }

  /// converts LocalPixel in ROC to DU coordinates. 
  /// LocalPixel must be inside ROC. Otherwise result is meaningless
  GlobalPixel toGlobal(const LocalPixel & loc) const {
    GlobalPixel result;
    result.col    = theFrameConverter.collumn().convert(loc.rocCol());
    result.row    = theFrameConverter.row().convert(loc.rocRow());
    return result;
  }

  /// printout for debug
  std::string print(int depth = 0) const;

  void initFrameConversion();

private:
  uint32_t theDetUnit;
  unsigned int theIdDU, theIdLk;
  FrameConversion theFrameConverter;

};

}

#endif
