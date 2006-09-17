#ifndef SiPixelDetId_PixelModuleName_H
#define SiPixelDetId_PixelModuleName_H

#include <string>
#include <iostream>

/** \class PixelModuleName
 * Base class to Pixel modules naming, provides a name as in PixelDatabase
 */

class PixelModuleName {
public:

  enum ModuleType { v1x2, v1x5, v1x8, v2x3, v2x4, v2x5, v2x8 } ;

  PixelModuleName(bool isBarrel) : barrel(isBarrel) { }
  virtual ~PixelModuleName() { }

  /// true for barrel modules
  virtual bool isBarrel() const { return barrel; }

  /// associated name 
  virtual std::string name() const = 0;

  /// module type
  virtual ModuleType moduleType() const = 0;

private:
  bool barrel;
};

std::ostream & operator<<( std::ostream& out, const PixelModuleName::ModuleType & t);
#endif
