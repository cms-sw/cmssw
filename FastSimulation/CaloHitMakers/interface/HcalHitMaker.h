#ifndef HcalHitMaker_h
#define HcalHitMaker_h

#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/CaloHitMakers/interface/CaloHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

//CLHEP headers
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Geometry/Transform3D.h"

#include <boost/cstdint.hpp>

#include <iostream>



class CaloGeometryHelper;

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
  const std::map<uint32_t,float>& getHits() { return hitMap_ ;} ;

  /// set the depth in X0 or Lambda0 units depending on showerType
  bool setDepth(double); 

 private:
    EcalHitMaker& myGrid;
    
    const FSimTrack * myTrack;
    HepPoint3D ecalEntrance_;
    HepVector3D particleDirection;
    int onHcal;
    
    double currentDepth_;
    HepTransform3D locToGlobal_;
    double radiusFactor_;
    bool mapCalculated_;
    
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
