#ifndef SiPixelDetId_PixelBarrelName_H
#define SiPixelDetId_PixelBarrelName_H

/** \class PixelBarrelName
 * Module name (as in PixelDatabase) in barrel
 */

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include <string>

class DetId; 

class PixelBarrelName : public PixelModuleName {
public:

  enum Shell { mO = 1, mI = 2 , pO =3 , pI =4 };

  /// ctor from DetId
  PixelBarrelName(const DetId &);

/*
  /// ctor for defined name
  PixelBarrelName(int layer, int module, int ladder)
    : PixelModuleName(true), 
      theLayer(layer), theModule(module), theLadder(ladder) { }
*/

  virtual ~PixelBarrelName() { }

  /// from base class
  virtual std::string name() const;

  Shell shell() const { return thePart; }
  
  int sectorName() const { return theSector; } 

  /// layer id 
  int layerName() const { return theLayer; }   

  /// module id (index in z) 
  int moduleName() const { return theModule; }  

  /// ladder id (index in phi) 
  int ladderName() const { return theLadder; } 

  /// full or half module
  bool isHalfModule() const { return halfModule; };
  

private:
  Shell thePart;
  bool halfModule;
  int theSector, theLayer, theModule, theLadder;
};

std::ostream & operator<<( std::ostream& out, const PixelBarrelName::Shell& t);
#endif
