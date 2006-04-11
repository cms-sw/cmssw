#ifndef RecoECAL_ECALClusters_PositionAwareHit_h
#define RecoECAL_ECALClusters_PositionAwareHit_h

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"


class PositionAwareHit
{

 private:

  // this position aware hit corresponds to a rec hit:
  EcalRecHit rechit;

  // The position vector of the hit crystal::
  GlobalPoint position;

  bool used;

 public:

  PositionAwareHit(EcalRecHit &the_rechit, const CaloSubdetectorGeometry &the_geometry);
  
  int operator<(const PositionAwareHit &other_hit) const;
  
  void use() { used = true; }
  bool isUsed() { return used; }
  
  double getEta() { return position.eta(); }
  double getTheta() { return position.theta(); }
  double getPhi() { return position.phi(); }
  double getEnergy() const { return rechit.energy(); }
  double getET() { double ET = getEnergy() * sin(getTheta()); return ET; }
  
  EBDetId getId() { return rechit.detid(); }
  EcalRecHit getHit() { return rechit; }
};

#endif
