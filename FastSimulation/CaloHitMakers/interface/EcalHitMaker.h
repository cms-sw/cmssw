#ifndef FastSimulation_CaloHitMakers_EcalHitMaker_h
#define FastSimulation_CaloHitMakers_EcalHitMaker_h

#include "Geometry/CaloTopology/interface/CaloDirection.h"

//#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/CaloHitMakers/interface/CaloHitMaker.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloPoint.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloSegment.h"
#include "FastSimulation/CaloGeometryTools/interface/CrystalPad.h"
#include "FastSimulation/CaloGeometryTools/interface/Crystal.h"
#include "FastSimulation/Utilities/interface/FamosDebug.h"

//#include <boost/cstdint.hpp>

#include <vector>

class CaloGeometryHelper;
class CrystalWindowMap;
class Histos;
class RandomEngine;
class FSimTrack;

class EcalHitMaker: public CaloHitMaker
{
 public:

  typedef math::XYZVector XYZVector;
  typedef math::XYZVector XYZPoint;
  typedef math::XYZVector XYZNormal;
  typedef ROOT::Math::Plane3D Plane3D;

  EcalHitMaker(CaloGeometryHelper * calo,
	       const XYZPoint& ecalentrance,
	       const DetId& cell,
	       int onEcal,
	       unsigned size,
	       unsigned showertype,
	       const RandomEngine* engine);

  ~EcalHitMaker();

  // This is not part of the constructor but it has to be called very early
  void setTrackParameters(const XYZNormal& normal,
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

  // number of X0 between PS2 and EE
  inline double ps2eeTotalX0() const {return X0PS2EE_;}
  
  /// in the ECAL 
  inline double ecalTotalX0() const {return X0ECAL_;}

  /// ECAL-HCAL transition 
  inline double ecalHcalGapTotalX0() const {  return X0EHGAP_;}

  /// in the HCAL 
  inline double hcalTotalX0() const {return X0HCAL_;}

  /// total number of L0 in the PS (Layer1). 
  inline double ps1TotalL0() const {return L0PS1_;}
  
  /// total number of L0 in the PS (Layer2). 
  inline double ps2TotalL0() const {return L0PS2_;}

  // number of X0 between PS2 and EE
  inline double ps2eeTotalL0() const {return L0PS2EE_;}
  
  /// in the ECAL 
  inline double ecalTotalL0() const {return L0ECAL_;}

  /// in the HCAL 
  inline double hcalTotalL0() const {return L0HCAL_;}

  /// ECAL-HCAL transition 
  inline double ecalHcalGapTotalL0() const {  return L0EHGAP_;}

  /// retrieve the segments (the path in the crystal crossed by the extrapolation
  /// of the track. Debugging only 
  inline const std::vector<CaloSegment>& getSegments() const {return segments_;};


  // The following methods are EM showers specific
  
  /// computes the crystals-plan intersection at depth (in X0 or L0 depending on the
  ///shower type)
  /// if it is not possible to go at such a depth, the result is false
  bool getPads(double depth,bool inCm=false) ;

  inline double getX0back() const {return maxX0_;}

  bool addHitDepth(double r,double phi,double depth=-1);
   
  bool addHit(double r,double phi,unsigned layer=0) ;

  unsigned fastInsideCell(const CLHEP::Hep2Vector & point,double & sp,bool debug=false) ;

  inline void setSpotEnergy(double e) { spotEnergy=e;}
  
   /// get the map of the stored hits. Triggers the calculation of the grid if it has
  /// not been done. 
    
  const std::map<CaloHitID,float>& getHits() ;
 
  /// To retrieve the track
  const FSimTrack* getFSimTrack() const {return myTrack_;}

  ///   used in FamosHcalHitMaker
  inline const XYZPoint& ecalEntrance() const {return EcalEntrance_;};

  inline void setRadiusFactor(double r) {radiusCorrectionFactor_ = r;}

  inline void setPulledPadSurvivalProbability(double val) {pulledPadProbability_ = val;};

  inline void setCrackPadSurvivalProbability(double val ) {crackPadProbability_ = val ;};

 // set preshower
 inline void setPreshowerPresent(bool ps) {simulatePreshower_=ps;};

  /// for debugging
  inline const std::vector<Crystal>& getCrystals() const {return regionOfInterest_;}



 private:



 // Computes the intersections of a track with the different calorimeters 
 void cellLine(std::vector<CaloPoint>& cp) ;

 void preshowerCellLine(std::vector<CaloPoint>& cp) const;

 void hcalCellLine(std::vector<CaloPoint>& cp) const;

 void ecalCellLine(const XYZPoint&, const XYZPoint&,std::vector<CaloPoint>& cp); 

 void buildSegments(const std::vector<CaloPoint>& cp);

 // retrieves the 7x7 crystals and builds the map of neighbours
 void buildGeometry();

 // depth-dependent geometry operations
 void configureGeometry();

 // project fPoint on the plane (origin,normal)
 bool pulled(const XYZPoint & origin, const XYZNormal& normal, XYZPoint & fPoint) const ;
 
 //  the numbering within the grid
 void prepareCrystalNumberArray();

 // find approximately the pad corresponding to (x,y)
 void convertIntegerCoordinates(double x, double y,unsigned &ix,unsigned &iy) const ;

 // pads reorganization (to lift the gaps)
 void reorganizePads();

 // retrieves the coordinates of a corner belonging to the neighbour
 typedef std::pair<CaloDirection,unsigned > neighbour;
 CLHEP::Hep2Vector & correspondingEdge(neighbour& myneighbour,CaloDirection dir2 ) ;

 // almost the same
 bool diagonalEdge(unsigned myPad, CaloDirection dir,CLHEP::Hep2Vector & point);

 // check if there is an unbalanced direction in the input vertor. If the result is true, 
 // the cooresponding directions are returned dir1+dir2=unb
 bool unbalancedDirection(const std::vector<neighbour>& dirs,unsigned & unb,unsigned & dir1, unsigned & dir2);

 // glue the pads together if there is no crack between them 
 void gapsLifting(std::vector<neighbour>& gaps,unsigned iq);

 // creates a crack
 void cracksPads(std::vector<neighbour> & cracks, unsigned iq);


 private:

 bool inside3D(const std::vector<XYZPoint>&, const XYZPoint& p) const;

  // the numbering of the pads
  std::vector<std::vector<unsigned > > myCrystalNumberArray_;

  // The following quantities are related to the path of the track through the detector
  double totalX0_;
  double totalL0_;
  double X0depthoffset_;
  double X0PS1_;
  double X0PS2_;
  double X0PS2EE_;
  double X0ECAL_;
  double X0EHGAP_;
  double X0HCAL_;
  double L0PS1_;
  double L0PS2_;
  double L0PS2EE_;
  double L0ECAL_;
  double L0HCAL_;
  double L0EHGAP_;
 
  double maxX0_;
  double rearleakage_ ;
  double outsideWindowEnergy_;

  // Grid construction 
  Crystal pivot_;
  XYZPoint EcalEntrance_;
  XYZNormal normal_;
  int central_;
  int onEcal_;

  bool configuredGeometry_ ;
  unsigned ncrystals_ ;
  // size of the grid in the(x,y) plane
  unsigned nx_,ny_;
  double xmin_,xmax_,ymin_,ymax_;

  std::vector<DetId> CellsWindow_;
  std::vector<Crystal> regionOfInterest_;
  std::vector<float> hits_;
  // Validity of the pads. To be valid, the intersection of the crytal with the plane should have 4 corners
  std::vector<bool> validPads_;
  // Get the index of the crystal (in hits_ or regionOfInterest_) when its CellID is known 
  // Needed because the navigation uses DetIds. 
  std::map<DetId,unsigned> DetIdMap_;

  CrystalWindowMap * myCrystalWindowMap_;

   // First segment in ECAL
  int ecalFirstSegment_;

  // Properties of the crystal window 
  unsigned etasize_;
  unsigned phisize_;
  // is the grid complete ? 
  bool truncatedGrid_ ;


  // shower simulation quantities
  // This one is the shower enlargment wrt Grindhammer
  double radiusCorrectionFactor_;
  // moliere radius  * radiuscorrectionfactor OR interactionlength
  double radiusFactor_ ; 
  // is it necessary to trigger the detailed simulation of the shower tail ? 
  bool detailedShowerTail_;
  // current depth
  double currentdepth_;
  // magnetic field correction factor
  double bfactor_;
  // simulate preshower
  bool simulatePreshower_;
  
  // pads-depth specific quantities 
  unsigned ncrackpadsatdepth_;
  unsigned npadsatdepth_;
  Plane3D plan_;
  // spot survival probability for a pulled pad - corresponding to the front face of a crystal
  // on the plan located in front of the crystal - Front leaking
  double pulledPadProbability_;
  // spot survival probability for the craks
  double crackPadProbability_;
  // size of the grid in the plane
  double sizex_,sizey_;  

 //  int fsimtrack_;
  const FSimTrack* myTrack_;
  
  // vector of the intersection of the track with the dectectors (PS,ECAL,HCAL)
  std::vector<CaloPoint> intersections_;
  // segments obtained from the intersections
  std::vector<CaloSegment> segments_;
  // should the pads be reorganized (most of the time YES! )
  bool doreorg_;

  // the geometrical objects
  std::vector<CrystalPad> padsatdepth_;
  std::vector<CrystalPad> crackpadsatdepth_;

  bool hitmaphasbeencalculated_ ;

  // A local vector of corners, to avoid reserving, newing and mallocing
  std::vector<CLHEP::Hep2Vector> mycorners;
  std::vector<XYZPoint> corners;


  const RandomEngine* random;


#ifdef FAMOSDEBUG
  Histos * myHistos;
#endif

  
};

#endif
