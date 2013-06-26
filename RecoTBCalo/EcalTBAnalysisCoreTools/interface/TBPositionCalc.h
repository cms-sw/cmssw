#ifndef TBPositionCalc_h
#define TBPositionCalc_h

#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "Rtypes.h"

#include <fstream>
#include <vector>
#include <cmath>
#include <map>

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalTestBeam/interface/EcalTBCrystalMap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h" 


class TBPositionCalc
{
 public:
  
  TBPositionCalc(const std::map<std::string,double>& providedParameters, const std::string& mapFile, const CaloSubdetectorGeometry *passedGeometry);  

  TBPositionCalc() { };

  ~TBPositionCalc();

  CLHEP::Hep3Vector CalculateTBPos(const std::vector<EBDetId>& passedDetIds, int myCrystal, EcalRecHitCollection const *passedRecHitsMap);
  
  CLHEP::Hep3Vector CalculateCMSPos(const std::vector<EBDetId>& passedDetIds, int myCrystal, EcalRecHitCollection const *passedRecHitsMap);
  
  void computeRotation(int myCrystal, CLHEP::HepRotation & CMStoTB );
    

 private:
  bool        param_LogWeighted_;
  Double32_t  param_X0_;
  Double32_t  param_T0_;
  Double32_t  param_W0_;

  EcalTBCrystalMap * theTestMap;

  const CaloSubdetectorGeometry *theGeometry_;
};
  
#endif


