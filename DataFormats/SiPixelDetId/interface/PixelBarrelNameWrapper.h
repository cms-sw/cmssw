#ifndef SiPixelDetId_PixelBarrelNameWrapper_H
#define SiPixelDetId_PixelBarrelNameWrapper_H

#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PixelBarrelNameWrapper {
public:
  
  PixelBarrelNameWrapper(const edm::ParameterSet&, const DetId&);
  
  virtual ~PixelBarrelNameWrapper() {}
  
  /// layer id
  int layerName() const {return pixelBarrelNameBase->layerName();}

  /// module id (index in z)
  int moduleName() const {return pixelBarrelNameBase->moduleName();}

  /// ladder id (index in phi)
  int ladderName() const {return pixelBarrelNameBase->ladderName();}

  /// sector id
  int sectorName() const {return pixelBarrelNameBase->sectorName();}

  /// full or half module
  bool isHalfModule() const {return pixelBarrelNameBase->isHalfModule();}

private:
  
  PixelBarrelNameBase* pixelBarrelNameBase;
  
  bool isUpgrade;

};

#endif
