#ifndef PreshowerHitMaker_h
#define PreshowerHitMaker_h

#include "FastSimulation/CaloHitMakers/interface/CaloHitMaker.h"
#include "FastSimulation/CaloGeometryTools/interface/Transform3DPJ.h"

class CaloGeometryHelper;
class LandauFluctuationGenerator;

class PreshowerHitMaker : public CaloHitMaker
{
 public:

  typedef math::XYZVector XYZVector;
  typedef math::XYZVector XYZPoint;
  typedef ROOT::Math::Transform3DPJ Transform3D;

  PreshowerHitMaker(CaloGeometryHelper * calo, 
		    const XYZPoint & , 
		    const XYZVector& , 
		    const XYZPoint& ,
		    const XYZVector&,
		    const LandauFluctuationGenerator* aGenerator);

  ~PreshowerHitMaker() {;} 
  
  inline void setSpotEnergy(double e) { spotEnergy=e;} 
  bool addHit(double r,double phi,unsigned layer=0);
  const std::map<unsigned,float>& getHits() { return hitMap_ ;} ;
 // for tuning
  inline void setMipEnergy(double e1, double e2) { mip1_=e1 ; mip2_=e2;} 


 private:

  XYZPoint psLayer1Entrance_;
  XYZVector psLayer1Dir_;
  XYZPoint psLayer2Entrance_;
  XYZVector psLayer2Dir_;
  bool layer1valid_;
  bool layer2valid_;
  Transform3D locToGlobal1_;
  Transform3D locToGlobal2_;
  float anglecorrection1_;
  float anglecorrection2_;
  double mip1_,mip2_;

  /// The Landau Fluctuation generator
  const LandauFluctuationGenerator*  theGenerator;

};

#endif
