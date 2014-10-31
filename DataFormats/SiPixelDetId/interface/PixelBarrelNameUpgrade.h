#ifndef SiPixelDetId_PixelBarrelNameUpgrade_H
#define SiPixelDetId_PixelBarrelNameUpgrade_H

/** \class PixelBarrelNameUpgrade
 * Module name (as in PixelDatabase) in barrel
 */

#include <string>
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameBase.h"

class DetId; 

class PixelBarrelNameUpgrade : public PixelModuleName, public PixelBarrelNameBase {
public:

  /// ctor from DetId
  PixelBarrelNameUpgrade(const DetId &);

  /// ctor for defined name with dummy parameters
  PixelBarrelNameUpgrade(Shell shell=mO, int layer=0, int module=0, int ladder=0)
    : PixelModuleName(true), 
      PixelBarrelNameBase(shell, layer, module, ladder) 
  { }

  /// ctor from name string
  PixelBarrelNameUpgrade(std::string name);

  virtual ~PixelBarrelNameUpgrade() { }

  /// from base class
  virtual std::string name() const;

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
};

#endif
