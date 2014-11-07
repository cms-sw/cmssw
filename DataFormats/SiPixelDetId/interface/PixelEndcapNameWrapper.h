#ifndef SiPixelDetId_PixelEndcapNameWrapper_H
#define SiPixelDetId_PixelEndcapNameWrapper_H

#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PixelEndcapNameWrapper {
public:
  
  PixelEndcapNameWrapper(const edm::ParameterSet&, const DetId&);
  
  virtual ~PixelEndcapNameWrapper() {}
  
  PixelEndcapNameBase::HalfCylinder halfCylinder() const { return pixelEndcapNameBase->halfCylinder(); }

  /// disk id
  int diskName() const { return pixelEndcapNameBase->diskName(); }

  /// blade id
  int bladeName() const { return pixelEndcapNameBase->bladeName(); }

  /// pannel id
  int pannelName() const { return pixelEndcapNameBase->pannelName(); }

  /// plaquetteId (in pannel)
  int plaquetteName() const { return pixelEndcapNameBase->plaquetteName(); }

private:
  
  PixelEndcapNameBase* pixelEndcapNameBase;
  
  bool isUpgrade;

};

#endif
