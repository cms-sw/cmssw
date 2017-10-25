#ifndef SiPixelDetId_PixelBarrelNameUpgrade_H
#define SiPixelDetId_PixelBarrelNameUpgrade_H

/** \class PixelBarrelNameUpgrade
 * Module name (as in PixelDatabase) in barrel
 */

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include <string>
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

class DetId; 

class PixelBarrelNameUpgrade : public PixelModuleName {
public:

  enum Shell { mO = 1, mI = 2 , pO =3 , pI =4 };

  /// ctor from DetId
  PixelBarrelNameUpgrade(const DetId &);

  /// ctor for defined name with dummy parameters
  PixelBarrelNameUpgrade(Shell shell=mO, int layer=0, int module=0, int ladder=0)
    : PixelModuleName(true), 
      thePart(shell), theLayer(layer), theModule(module), theLadder(ladder) 
  { }

  /// ctor from name string
  PixelBarrelNameUpgrade(std::string name);

  ~PixelBarrelNameUpgrade() override { }

  /// from base class
  std::string name() const override;

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
  PixelModuleName::ModuleType  moduleType() const override;

  /// return the DetId
  PXBDetId getDetId();

  /// check equality of modules from datamemebers
  bool operator== (const PixelModuleName &) const override;

private:
  Shell thePart;
  int theLayer, theModule, theLadder;
};

std::ostream & operator<<( std::ostream& out, const PixelBarrelNameUpgrade::Shell& t);
#endif
