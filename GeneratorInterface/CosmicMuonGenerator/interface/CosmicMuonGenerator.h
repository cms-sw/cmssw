#ifndef CosmicMuonGenerator_h
#define CosmicMuonGenerator_h
//
// CosmicMuonGenerator by droll (04/DEC/2005)
// modified by P. Biallass 29.03.2006 to implement new cosmic generator (CMSCGEN.cc) 
//

// include files
#include <iostream>

#include "GeneratorInterface/CosmicMuonGenerator/interface/CMSCGENnorm.h"
#include "GeneratorInterface/CosmicMuonGenerator/interface/CMSCGEN.h"
#include "GeneratorInterface/CosmicMuonGenerator/interface/CosmicMuonParameters.h"
#include "GeneratorInterface/CosmicMuonGenerator/interface/SingleParticleEvent.h"

namespace CLHEP {
  class HepRandomEngine;
}

// class definitions
class CosmicMuonGenerator{
public:
  // constructor
  CosmicMuonGenerator();
  // destructor
  ~CosmicMuonGenerator();
  // event with one particle
  //SingleParticleEvent OneMuoEvt;
  SingleParticleEvent OneMuoEvt;
 
 

private:
  //initialize class which normalizes flux (added by P.Biallass 29.3.2006)
  CMSCGENnorm*  Norm ;
  //initialize class which produces the cosmic muons  (modified by P.Biallass 29.3.2006)
  CMSCGEN* Cosmics ; 
  // default control parameters
  unsigned int NumberOfEvents; // number of events to be generated
  int    RanSeed; // seed of random number generator
  double MinP;     // min. E     [GeV]
  double MinP_CMS; // min. E at CMS surface    [GeV]; default is MinE_CMS=MinE, thus no bias from access-shaft
  double MaxP;     // max. E     [GeV]
  double MinTheta; // min. theta [rad]
  double MaxTheta; // max. theta [rad]
  double MinPhi;   // min. phi   [rad]
  double MaxPhi;   // max. phi   [rad]
  double MinT0;    // min. t0   [ns]
  double MaxT0;    // max. t0   [ns]
  double ElossScaleFactor; // scale factor for energy loss
  double RadiusOfTarget; // Radius of target-cylinder which cosmics HAVE to hit [mm], default is CMS-dimensions
  double ZDistOfTarget; // z-length of target-cylinder which cosmics HAVE to hit [mm], default is CMS-dimensions
  double ZCentrOfTarget; // z-position of centre of target-cylinder which cosmics HAVE to hit [mm], default is Nominal Interaction Point (=0)
  bool   TrackerOnly; //if set to "true" detector with tracker-only setup is used, so no material or B-field outside is considerd
  bool   TIFOnly_constant; //if set to "true" cosmics can also be generated below 2GeV with unphysical constant energy dependence
  bool   TIFOnly_linear; //if set to "true" cosmics can also be generated below 2GeV with unphysical linear energy dependence
  bool   MTCCHalf; //if set to "true" muons are sure to hit half of CMS important for MTCC, 
                   //still material and B-field of whole CMS is considered
  double EventRate; // number of muons per second [Hz]
  double rateErr_stat; // stat. error of number of muons per second [Hz]
  double rateErr_syst; // syst. error of number of muons per second [Hz] from error of known flux
  // other stuff needed
  double SumIntegrals; // sum of phase space integrals
  double Ngen; // number of generated events
  double Nsel; // number of selected events
  double Ndiced; // number of diced events
  double Target3dRadius; // radius of sphere around target (cylinder)
  double SurfaceRadius; // radius for area on surface that has to be considered (for event generation)
  double PlugVx; //Plug x position
  double PlugVz; //Plug z position

  //For upgoing muon generation: Neutrino energy limits
  double MinEnu;
  double MaxEnu;

  bool AcptAllMu; //Accepting All Muons regardeless of direction


  // random number generator
  CLHEP::HepRandomEngine *RanGen;
  bool delRanGen;
  // check user input
  bool NotInitialized;
  void checkIn();
  // check, if muon is pointing into target
  bool goodOrientation();
  // event display: initialize + display
  void initEvDis();
  void displayEv();

public:
  // set parameters
  void setNumberOfEvents(unsigned int N);
  void setRanSeed(int N);
  void setMinP(double P);
  void setMinP_CMS(double P);
  void setMaxP(double P);
  void setMinTheta(double Theta);
  void setMaxTheta(double Theta);
  void setMinPhi(double Phi);
  void setMaxPhi(double Phi);
  void setMinT0(double T0);
  void setMaxT0(double T0);
  void setElossScaleFactor(double ElossScaleFact);
  void setRadiusOfTarget(double R);
  void setZDistOfTarget(double Z);
  void setZCentrOfTarget(double Z);
  void setTrackerOnly(bool Tracker);
  void setTIFOnly_constant(bool TIF);
  void setTIFOnly_linear(bool TIF);
  void setMTCCHalf(bool MTCC);
  void setPlugVx(double PlugVtx);
  void setPlugVz(double PlugVtz);
  void setMinEnu(double MinEn);
  void setMaxEnu(double MaxEn);
  void setAcptAllMu(bool AllMu);


  // initialize the generator
  void initialize(CLHEP::HepRandomEngine *rng = 0);
   // prints rate + statistics
  void terminate();
  // initialize, generate and terminate the Cosmic Muon Generator
  void runCMG();
  // returns event rate
  double getRate();
  // generate next event/muon
  void nextEvent();
};
#endif
