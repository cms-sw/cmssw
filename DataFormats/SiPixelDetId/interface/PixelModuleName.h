#ifndef SiPixelDetId_PixelModuleName_H
#define SiPixelDetId_PixelModuleName_H

#include <string>

/** \class PixelModuleName
 * Base class to Pixel modules naming, provides a name as in PixelDatabase
 */

class PixelModuleName {
public:
  PixelModuleName(bool isBarrel) : barrel(isBarrel) { }
  virtual ~PixelModuleName() { }

  /// true for barrel modules
  virtual bool isBarrel() const { return barrel; }

  /// associated name 
  virtual std::string name() const = 0;

private:
  bool barrel;
};
#endif
