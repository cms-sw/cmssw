#ifndef EcalHitMaker_h
#define EcalHitMaker_h

#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/CaloHitMakers/interface/CaloHitMaker.h"

// CLHEP headers
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Plane3D.h"

class Calorimeter;

class EcalHitMaker: public CaloHitMaker
{
 public:
  EcalHitMaker(Calorimeter * calo,
	       const HepPoint3D& ecalentrance,
	       const DetId& cell,
	       int onEcal,
	       unsigned size,
	       unsigned showertype);
  ~EcalHitMaker();

  // This is not part of the constructor but it has to be called very early
  void setTrackParameters(const HepNormal3D& normal,
			  double X0depthoffset,
			  const FSimTrack& theTrack);
  
  // The following methods are related to the path of the particle
  // through the detector. 

  // Number of X0 "seen" by the track 
    //  inline double totalX0() const {return totalX0_-X0depthoffset_;}; 
  inline double totalX0() const {return totalX0_;}; 

    /// Number of interaction length "seen" by the track 
  inline double totalL0() const {return totalL0_;}; 

  /// get the offset (e.g the number of X0 after which the shower starts)
  inline double x0DepthOffset() const {return X0depthoffset_;}

  // total number of X0 in the PS (Layer1). 
  inline double ps1TotalX0() const {return X0PS1_;}
  
  /// total number of X0 in the PS (Layer2). 
  inline double ps2TotalX0() const {return X0PS2_;}
  
  /// in the ECAL 
  inline double ecalTotalX0() const {return X0ECAL_;}

  /// ECAL-HCAL transition 
  inline double ecalHcalGapTotalX0() const {  return X0EHGAP_;}

  /// in the HCAL 
  inline double hcalTotalX0() const {return X0HCAL_;}

  /// total number of L0 in the PS (Layer1). 
  inline double ps1TotalL0() const {return L0PS1_;}
  
  /// total number of L0 in the PS (Layer1). 
  inline double ps2TotalL0() const {return L0PS2_;}
  
  /// in the ECAL 
  inline double ecalTotalL0() const {return L0ECAL_;}

  /// in the HCAL 
  inline double hcalTotalL0() const {return L0HCAL_;}

  /// ECAL-HCAL transition 
  inline double ecalHcalGapTotalL0() const {  return L0EHGAP_;}


  // The following methods are EM showers specific
  
  /// computes the crystals-plan intersection at depth (in X0)
  /// if it is not possible to go at such a depth, the result is false
  bool getQuads(double depth) ;

  inline double getX0back() const {return maxX0_;}

  bool addHitDepth(double r,double phi,double depth=-1);
   
  // must be implemented 
  bool addHit(double r,double phi,unsigned layer=0) ;

  
  inline void setSpotEnergy(double e) { spotEnergy=e;}
  
// get the map of the stored hits. Triggers the calculation of the grid if it has
  /// not been done. 
  const std::map<unsigned,float>& getHits() {return hitMap_;} 
 
  // To retrieve the track
  const FSimTrack*getFSimTrack() const {return myTrack_;}

  //   used in FamosHcalHitMaker
  inline const HepPoint3D& ecalEntrance() const {return EcalEntrance_;};

 private:

  // The following quantities are related to the path of the track through the detector
  double totalX0_;
  double totalL0_;
  double X0depthoffset_;
  double X0PS1_;
  double X0PS2_;
  double X0ECAL_;
  double X0EHGAP_;
  double X0HCAL_;
  double L0PS1_;
  double L0PS2_;
  double L0ECAL_;
  double L0HCAL_;
  double L0EHGAP_;
 
  double maxX0_;
 
  DetId pivot_;
  HepPoint3D EcalEntrance_;
  HepNormal3D normal_;
 
 //  int fsimtrack_;
  const FSimTrack* myTrack_;
  
  
};

#endif
