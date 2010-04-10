#ifndef PreshowerHitMaker_h
#define PreshowerHitMaker_h

#include "FastSimulation/CaloHitMakers/interface/CaloHitMaker.h"

class CaloGeometryHelper;
class LandauFluctuationGenerator;

class PreshowerHitMaker : public CaloHitMaker
{
 public:

  typedef math::XYZVector XYZVector;
  typedef math::XYZVector XYZPoint;

  PreshowerHitMaker(CaloGeometryHelper * calo, 
		    const XYZPoint & , 
		    const XYZVector& ,
		    const XYZPoint& ,
		    const XYZVector&,
		    const LandauFluctuationGenerator* aGenerator);

  ~PreshowerHitMaker() {;}
  // currently deprecated 
  inline void setSpotEnergy(double e) { spotEnergy=e;} 
  // for tuning
  inline void setMipEnergy(double e1, double e2) { mip1_=e1 ; mip2_=e2;} 
  bool addHit(double r,double phi,unsigned layer=0);
  const std::map<unsigned,float>& getHits() { return hitMap_ ;} ;


 private:

  XYZPoint psLayer1Entrance_;
  XYZVector psLayer1Dir_;
  XYZPoint psLayer2Entrance_;
  XYZVector psLayer2Dir_;
  bool layer1valid_;
  bool layer2valid_;
  double invcostheta1x;
  double invcostheta1y;
  double invcostheta2x;
  double invcostheta2y;
  double x1,y1,z1;
  double x2,y2,z2;
  double mip1_,mip2_;

  /// The Landau Fluctuation generator
  const LandauFluctuationGenerator*  theGenerator;

};

#endif
