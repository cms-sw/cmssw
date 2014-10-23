#ifndef SiPixelDetId_PixelBarrelNameBase_H
#define SiPixelDetId_PixelBarrelNameBase_H

class PixelBarrelNameBase {
public:

  /// layer id
  virtual int layerName() const = 0;

  /// module id (index in z)
  virtual int moduleName() const = 0;

  /// ladder id (index in phi)
  virtual int ladderName() const = 0;

  /// sector id
  virtual int sectorName() const = 0;

  /// full or half module
  virtual bool isHalfModule() const = 0;

};

#endif
