#ifndef PixelBarrelName_H
#define PixelBarrelName_H

/** \class PixelBarrelName
 * Module name (as in PixelDatabase) in barrel
 */

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include <string>

class DetId; 

class PixelBarrelName : public PixelModuleName {
public:

  /// ctor from DetId
  PixelBarrelName(const DetId &);

  /// ctor for defined name
  PixelBarrelName(int layer, int module, int ladder)
    : PixelModuleName(true), 
      theLayer(layer), theModule(module), theLadder(ladder) { }

  virtual ~PixelBarrelName() { }

  /// from base class
  virtual std::string name() const;

  /// layer id 
  int layerName() const { return theLayer; }   

  /// module id (index in z) 
  int moduleName() const { return theModule; }  

  /// ladder id (index in phi) 
  int ladderName() const { return theLadder; } 

  /// full or half module
  bool isFullModule() const;
  

private:
  int theLayer, theModule, theLadder;
};
#endif
