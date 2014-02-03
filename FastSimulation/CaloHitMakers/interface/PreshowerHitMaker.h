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

  const std::map<CaloHitID,float>& getHits() { return hitMap_ ;} ;
 // for tuning
  inline void setMipEnergy(double e1, double e2) { mip1_=e1 ; mip2_=e2;} 
  
  float totalLayer1() const { return totalLayer1_;}
  float totalLayer2() const { return totalLayer2_;}
  float layer1Calibrated() const { return 0.024/81.1E-6*totalLayer1_;}
  float layer2Calibrated() const { return 0.024*0.7/81.1E-6*totalLayer2_;}
  float totalCalibrated() const { return 0.024/81.1E-6*(totalLayer1_+0.7*totalLayer2_);}

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
  float totalLayer1_;
  float totalLayer2_;
  /// The Landau Fluctuation generator
  const LandauFluctuationGenerator*  theGenerator;

};

#endif
