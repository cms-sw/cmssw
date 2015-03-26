#ifndef HDShower_H
#define HDShower_H

//FastSimulation Headers
#include "FastSimulation/ShowerDevelopment/interface/HDShowerParametrization.h"

#include "DataFormats/Math/interface/Vector3D.h"

#include <vector>

/** 
 * \author Salavat Abdullin
 * \date: 21-Oct-2004
 * \parameterized hadronic shower simulation 
 * \based on paper G.Gridhammer, M.Rudowicz, S.Peters, SLAC-PUB-5072
 */ 

class EcalHitMaker;
class HcalHitMaker;
class RandomEngineAndDistribution;

class HDShower 
{

 public:

  typedef math::XYZVector XYZPoint;

  typedef std::pair<XYZPoint,double> Spot;
  typedef std::pair<unsigned int, double> Step;
  typedef std::vector<Step> Steps;
  typedef Steps::const_iterator step_iterator;

  HDShower(const RandomEngineAndDistribution* engine,
	   HDShowerParametrization* myParam,
	   EcalHitMaker* myGrid, 
	   HcalHitMaker* myHcalHitMaker,
	   int onECAL, 
	   double epart,
	   double pmip);

  int getmip() {return mip;}

  virtual ~HDShower() {;}

  /// Compute the shower longitudinal and lateral development
  bool compute();

 private:

  // The longitudinal development ersatzt.
  double gam(double x, double a) const { return pow(x,a-1.)*exp(-x); }

  // Transverse integrated probability function (for 4R max size)
  // integral of the transverse ansatz f(r) =  2rR/(r**2 + R**2)**2 ->
  // 1/R - R/(R**2+r**2) | at limits 4R - 0
  double transProb(double factor, double R, double r) {
    double fsq = factor * factor; 
    return ((fsq + 1.)/fsq) * r * r / (r*r + R*R) ; 
  }
  // Compute longE[i] and transR[i] for all nsteps
  void makeSteps(int nsteps);

  int indexFinder(double x, const std::vector<double> & Fhist);  

  // The parametrization
  HDShowerParametrization* theParam;

  // The Calorimeter properties
  const ECALProperties* theECALproperties;
  const HCALProperties* theHCALproperties;

  // Basic parameters of the simulation
  double theR1, theR2, theR3;
  double alpEM, betEM, alpHD, betHD, part, tgamEM, tgamHD;

  // The basic quantities for the shower development.
  double lambdaEM, lambdaHD, x0EM, x0HD;
  double depthStart;
  double aloge;

  std::vector<int>    detector, nspots;
  std::vector<double> eStep, rlamStep;
  std::vector<double> x0curr, x0depth;
  std::vector<double> lamstep, lamcurr, lamdepth, lamtotal;
  
  int infinity; // big number of cycles if exit is on a special condition

  // The crystal grid
  EcalHitMaker * theGrid;

  // The HCAL 
  HcalHitMaker * theHcalHitMaker;

  // OnECAL flag as an input parameter ... 
  int onEcal;

  // MIP in ECAL map flag
  int mip;

  // Input energy to distribute
  double e;

  // HCAL losses option (0-off, 1-on)
  int lossesOpt;
  // Number of longitudinal steps in HCAL
  int nDepthSteps;
  // Number of bins in the transverse probability histogram
  int nTRsteps;
  // Transverse size tuning factor 
  double transParam;
  // Transverse normalization : 1 for HB/HE, 0.5 for HF (narrow showers)
  double transFactor;
  // HCAL energy spot size
  double eSpotSize;
  // Longitudinal step size (lambda units)
  double depthStep;
  // Energy threshold (one depth step -> nDepthSteps);
  double criticalEnergy;
  // Transverse size cut (in character transverse size units)
  double maxTRfactor;
  // Balance between ECAL and HCAL "visible" energy (default = 1.)
  double balanceEH;
  // Regulator of HCAL depth of the shower (to adjust/shrink it to CMS depth) 
  double hcalDepthFactor;

  // Famos Random Engine
  const RandomEngineAndDistribution* random;
  
  //calorimeter depths
  double depthECAL, depthGAP, depthGAPx0, depthHCAL, depthToHCAL;
};

#endif
