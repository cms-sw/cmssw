#ifndef PreshowerHitMaker_h
#define PreshowerHitMaker_h

#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/CaloHitMakers/interface/CaloHitMaker.h"
#include "FastSimulation/MaterialEffects/interface/LandauFluctuationGenerator.h"

//CLHEP headers
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Vector3D.h"

class CaloGeometryHelper;

class PreshowerHitMaker : public CaloHitMaker
{
 public:
  PreshowerHitMaker(CaloGeometryHelper * calo, const HepPoint3D & , const HepVector3D& ,const HepPoint3D& ,const HepVector3D& );
  ~PreshowerHitMaker() {;}
  
  inline void setSpotEnergy(double e) { spotEnergy=e;} 
  bool addHit(double r,double phi,unsigned layer=0);
  const std::map<unsigned,float>& getHits() { return hitMap_ ;} ;


  private:
  /// The Landau Fluctuation generator
  static LandauFluctuationGenerator theGenerator;

  HepPoint3D psLayer1Entrance_;
  HepVector3D psLayer1Dir_;
  HepPoint3D psLayer2Entrance_;
  HepVector3D psLayer2Dir_;
  double invcostheta1x;
  double invcostheta1y;
  double invcostheta2x;
  double invcostheta2y;
  double x1,y1,z1;
  double x2,y2,z2;
};

#endif
