#ifndef SiPixelDetId_PixelBarrelName_H
#define SiPixelDetId_PixelBarrelName_H

/** \class PixelBarrelName
 * Module name (as in PixelDatabase) in barrel
 */

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include <string>
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

class DetId; 

class PixelBarrelName : public PixelModuleName {
public:

  enum Shell { mO = 1, mI = 2 , pO =3 , pI =4 };

  /// ctor from DetId
  PixelBarrelName(const DetId &);

  /// ctor for defined name with dummy parameters
  PixelBarrelName(Shell shell=mO, int layer=0, int module=0, int ladder=0)
    : PixelModuleName(true), 
      thePart(shell), theLayer(layer), theModule(module), theLadder(ladder) 
  { }

  /// ctor from name string
  PixelBarrelName(std::string name);

  virtual ~PixelBarrelName() { }

  /// from base class
  virtual std::string name() const;

  Shell shell() const { return thePart; }

  /// layer id 
  int layerName() const { return theLayer; }   

  /// module id (index in z) 
  int moduleName() const { return theModule; }  

  /// ladder id (index in phi) 
  int ladderName() const { return theLadder; } 

  /// sector id
  int sectorName() const;

  /// full or half module
  bool isHalfModule() const;
  
  /// module Type
  virtual PixelModuleName::ModuleType  moduleType() const;

  /// return the DetId
  PXBDetId getDetId();

  /// check equality of modules from datamemebers
  virtual bool operator== (const PixelModuleName &) const;

private:
  Shell thePart;
  int theLayer, theModule, theLadder;
};

std::ostream & operator<<( std::ostream& out, const PixelBarrelName::Shell& t);
#endif
