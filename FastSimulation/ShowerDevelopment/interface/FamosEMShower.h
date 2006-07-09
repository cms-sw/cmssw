#ifndef FamosEMShower_H
#define FamosEMShower_H

#include "FastSimulation/Particle/interface/RawParticle.h"

//Famos Headers
#include "FastSimulation/ShowerDevelopment/interface/EMECALShowerParametrization.h"
#include "FastSimulation/ShowerDevelopment/interface/RadialInterval.h"
#include "FastSimulation/Utilitiess/interface/GammaFunctionGenerator.h"
#include "FastSimulation/Utilitiess/interface/Histos.h"
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/GenericFunctions/IncompleteGamma.hh"

typedef pair<HepPoint3D,double> Spot;
typedef pair<unsigned int, double> Step;
typedef vector<Step> Steps;
typedef Steps::const_iterator step_iterator;

/** 
 * \author Patrick Janot
 * \date: 25-Jan-2004
 */ 

class FamosGrid;
class FamosPreshower;
class FamosHcalHitMaker;
class GammaDistributionGenerator;
class FamosEMShower 
{

 public:

  FamosEMShower(EMECALShowerParametrization* const myParam,
	      vector<const RawParticle*>* const myPart,
	      FamosGrid* const myGrid=NULL,FamosPreshower * const myPreshower=NULL);

  virtual ~FamosEMShower(){;}

  /// Compute the shower longitudinal and lateral development
  void compute();

  /// get the depth of the centre of gravity of the shower(s)
  inline double getMeanDepth() const {return globalMeanDepth;};  

  /// set the grid address
  void setGrid(FamosGrid * const myGrid) { theGrid=myGrid;}

  /// set the preshower address
  void setPreshower(FamosPreshower * const myPresh ) ;

  /// set the HCAL address
  void setHcal(FamosHcalHitMaker * const myHcal);

 private:

  // The longitudinal development ersatzt.
  double gam(double x, double a) const;

  // Energy deposited in the layer t-dt-> t, in units of E0 (initial energy)
  double deposit(double t, double a, double b, double dt);

  // Energy deposited between 0 and t, in units of E0 (initial energy)
  double deposit(double a, double b, double t);

  // Set the intervals for the radial development
  void setIntervals(unsigned icomp,RadialInterval& rad);
  
  // The parametrization
  EMECALShowerParametrization* const theParam;

  // The Calorimeter properties
  const ECALProperties* theECAL;
  const HCALProperties* theHCAL;
  const PreshowerLayer1Properties* theLayer1;
  const PreshowerLayer2Properties* theLayer2;

  // The incident particle(s)
  vector<const RawParticle*>* const thePart;
  unsigned int nPart;

  // The basic quantities for the shower development.
  vector<double> theNumberOfSpots;
  vector<double> Etot;
  vector<double> E; 
  vector<double> photos;
  vector<double> T;
  vector<double> a; 
  vector<double> b;
  vector<double> Ti; 
  vector<double> TSpot;
  vector<double> aSpot; 
  vector<double> bSpot;

  vector<double> meanDepth;
  double globalMeanDepth;
  double totalEnergy;

  // The steps for the longitudinal development
  Steps steps; 

  // The crystal grid
  FamosGrid * theGrid;

  // The preshower 
  FamosPreshower * thePreshower;

  // The HCAL hitmaker
  FamosHcalHitMaker * theHcalHitMaker;

  // Is there a preshower ? 
  bool hasPreshower;
  // Histos
  //  Histos* myHistos;

  // integer gamma function generator
  GammaFunctionGenerator * myGammaGenerator;

  Genfun::IncompleteGamma myIncompleteGamma;
  
  
};

#endif
