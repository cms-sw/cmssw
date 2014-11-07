#ifndef SiPixelDetId_PixelBarrelNameBase_H
#define SiPixelDetId_PixelBarrelNameBase_H

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"

class PixelBarrelNameBase {
public:

  enum Shell { mO = 1, mI = 2 , pO =3 , pI =4 };

  /// ctor for defined name with dummy parameters
  PixelBarrelNameBase(Shell shell=mO, int layer=0, int module=0, int ladder=0, bool phase=false) :
    thePart(shell), theLayer(layer), theModule(module), theLadder(ladder)
  { }

  Shell shell() const { return thePart; }

  /// layer id
  int layerName() const { return theLayer; }

  /// module id (index in z)
  int moduleName() const { return theModule; }

  /// ladder id (index in phi)
  int ladderName() const { return theLadder; }

  /// full or half module
  virtual bool isHalfModule() const = 0;

protected:
  Shell thePart;
  int theLayer, theModule, theLadder;

};

std::ostream & operator<<( std::ostream& out, const PixelBarrelNameBase::Shell& t);
#endif
