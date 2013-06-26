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
  PixelROC() : theDetUnit(0), theIdDU(0), theIdLk(0), theFrameConverter(0) {} 

  ~PixelROC();

  PixelROC(const PixelROC & o);
  const PixelROC& operator=(const PixelROC&);

  void swap(PixelROC&);

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
  LocalPixel  toLocal(const GlobalPixel & gp) const;

  /// converts LocalPixel in ROC to DU coordinates. 
  /// LocalPixel must be inside ROC. Otherwise result is meaningless
  GlobalPixel toGlobal(const LocalPixel & loc) const;

  /// printout for debug
  std::string print(int depth = 0) const;

private:
  void initFrameConversion() const;

private:
  uint32_t theDetUnit;
  unsigned int theIdDU, theIdLk;
  mutable const FrameConversion * theFrameConverter;

};

}

#endif
