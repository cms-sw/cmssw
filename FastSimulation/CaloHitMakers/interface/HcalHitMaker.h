#ifndef HcalHitMaker_h
#define HcalHitMaker_h

#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/CaloHitMakers/interface/CaloHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

//CLHEP headers
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include <iostream>

class Calorimeter;

class HcalHitMaker : public CaloHitMaker
{
 public:
  HcalHitMaker(EcalHitMaker &, unsigned );
  ~HcalHitMaker() {;}
  
  /// Set the spot energy
  inline void setSpotEnergy(double e) { spotEnergy=e;} 
  
  /// add the hit in the HCAL
  bool addHit(double r,double phi,unsigned layer=0);
  
   /// get the hits
  const std::map<unsigned,float>& getHits() { return hitMap_ ;} ;

  /// set the depth in X0 or Lambda0 units depending on showerType
  bool setDepth(double); 

 private:
    EcalHitMaker& myGrid;
    
    const FSimTrack * myTrack;
    HepPoint3D ecalEntrance;
    HepVector3D particleDirection;
    int onHcal;
    
    unsigned showerType;
    double currentDepth;
    HepTransform3D locToGlobal;
    double radiusFactor;
    bool mapCalculated;
    
 public:
    static int getSubHcalDet(const FSimTrack* t)
      {
	//	std::cout << " getSubHcalDet " << std::endl;
	// According to  DataFormats/ HcalDetId/ interface/ HcalSubdetector.h
	//	std::cout << " onHcal " << t->onHcal() << " onVFcal " << t->onVFcal() << std::endl;
	if(t->onHcal()==1) return HcalBarrel;
	if(t->onHcal()==2) return HcalEndcap;
	if(t->onVFcal()==2) return HcalForward;
	return -1;
      } 

};
#endif
