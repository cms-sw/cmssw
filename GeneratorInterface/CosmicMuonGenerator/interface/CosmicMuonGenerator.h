#ifndef CosmicMuonGenerator_h
#define CosmicMuonGenerator_h
//
// CosmicMuonGenerator by droll (04/DEC/2005)
// modified by P. Biallass 29.03.2006 to implement new cosmic generator (CMSCGEN.cc) 
//

// include files

#include <CLHEP/Random/RandomEngine.h>
#include <CLHEP/Random/JamesRandom.h>

namespace CLHEP {
  class HepRandomEngine;
}

#include <iostream>
#include <string>
#include <vector>
#include "TFile.h"
#include "TTree.h"

#include "GeneratorInterface/CosmicMuonGenerator/interface/sim.h"

#include "GeneratorInterface/CosmicMuonGenerator/interface/CMSCGENnorm.h"
#include "GeneratorInterface/CosmicMuonGenerator/interface/CMSCGEN.h"
#include "GeneratorInterface/CosmicMuonGenerator/interface/CosmicMuonParameters.h"
#include "GeneratorInterface/CosmicMuonGenerator/interface/SingleParticleEvent.h"

// class definitions
class CosmicMuonGenerator{
public:
  // constructor
  CosmicMuonGenerator() : delRanGen(false)
    {
      //initialize class which normalizes flux (added by P.Biallass 29.3.2006)
      Norm = new CMSCGENnorm();
      //initialize class which produces the cosmic muons  (modified by P.Biallass 29.3.2006)
      Cosmics = new CMSCGEN();
      // set default control parameters
      NumberOfEvents = 100;
      RanSeed = 135799468;
      MinP =     3.;
      MinP_CMS =     MinP;
      MaxP =   3000.;
      MinTheta =  0.*Deg2Rad;
      //MaxTheta = 84.26*Deg2Rad;
      MaxTheta = 89.0*Deg2Rad;
      MinPhi =    0.*Deg2Rad;
      MaxPhi =  360.*Deg2Rad;
      MinT0  = -12.5;
      MaxT0  =  12.5;
      ElossScaleFactor = 1.0;
      RadiusOfTarget = 8000.;
      ZDistOfTarget = 15000.;
      ZCentrOfTarget = 0.;
      TrackerOnly = false;
      MultiMuon = false;
      MultiMuonFileName = "dummy.root";
      MultiMuonFileFirstEvent = 0;
      MultiMuonNmin = 2;
      TIFOnly_constant = false;
      TIFOnly_linear = false;
      MTCCHalf = false;
      EventRate = 0.;
      rateErr_stat = 0.;
      rateErr_syst = 0.;
      
      SumIntegrals = 0.;
      Ngen = 0.;
      Nsel = 0.;
      Ndiced = 0.;
      NotInitialized = true;
      Target3dRadius = 0.;
      SurfaceRadius = 0.;
      //set plug as default onto PX56 shaft
      PlugVx = PlugOnShaftVx;
      PlugVz = PlugOnShaftVz;
      //material densities in g/cm^3
      RhoAir = 0.001214;
      RhoWall = 2.5;
      RhoRock = 2.5;
      RhoClay = 2.3;
      RhoPlug = 2.5;
      ClayWidth = 50000; //[mm]


      
      std::cout << std::endl;
      std::cout << "*********************************************************" << std::endl;
      std::cout << "*********************************************************" << std::endl;
      std::cout << "***                                                   ***" << std::endl;
      std::cout << "***  C O S M I C  M U O N  G E N E R A T O R  (vC++)  ***" << std::endl;
      std::cout << "***                                                   ***" << std::endl;
      std::cout << "*********************************************************" << std::endl;
      std::cout << "*********************************************************" << std::endl;
      std::cout << std::endl;
    }
  
  // destructor
  ~CosmicMuonGenerator()
    {
      if (delRanGen)
	delete RanGen;
      delete Norm; 
      delete Cosmics;
    }
  
  // event with one particle
  //SingleParticleEvent OneMuoEvt;
  SingleParticleEvent OneMuoEvt;

  double EventWeight; //for multi muon events
  double Trials; //for multi muon events

  int Id_at;
  double Px_at; double Py_at; double Pz_at; 
  double E_at; 
  //double M_at;
  double Vx_at; double Vy_at; double Vz_at; 
  double T0_at;
  double Theta_at;


  std::vector<double> Px_mu; std::vector<double> Py_mu; std::vector<double> Pz_mu;
  std::vector<double> P_mu;
  std::vector<double> Vx_mu; std::vector<double> Vy_mu; std::vector<double> Vz_mu;
  double Vxz_mu;
  std::vector<double> Theta_mu;

  std::vector<int> Id_sf;
  std::vector<double> Px_sf; std::vector<double> Py_sf; std::vector<double> Pz_sf; 
  std::vector<double> E_sf; 
  //std::vector<double> M_sf;
  std::vector<double> Vx_sf; std::vector<double> Vy_sf; std::vector<double> Vz_sf; 
  std::vector<double> T0_sf;
  
  std::vector<int> Id_ug;
  std::vector<double> Px_ug; std::vector<double> Py_ug; std::vector<double> Pz_ug;
  std::vector<double> E_ug; 
  //std::vector<double> M_ug;
  std::vector<double> Vx_ug; std::vector<double> Vy_ug; std::vector<double> Vz_ug;
  std::vector<double> T0_ug;
 
 

private:

  TFile* MultiIn; //file to be read in
  TTree* MultiTree; //tree of file with multi muon events
  sim* SimTree; //class to acces tree branches
  ULong64_t SimTreeEntries;
  ULong64_t SimTree_jentry;
  int NcloseMultiMuonEvents;
  int NskippedMultiMuonEvents;


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
  bool   MultiMuon; //read in multi-muon events from file instead of generating single muon events
  std::string MultiMuonFileName; //file containing multi muon events, to be read in
  int MultiMuonFileFirstEvent; //first multi muon event, to be read in
  int MultiMuonNmin; //minimal number of multi muons per event reaching the cylinder surrounding CMS
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

  //material densities in g/cm^3
  double RhoAir;
  double RhoWall;
  double RhoRock;
  double RhoClay;
  double RhoPlug;
  double ClayWidth; //[mm]


  //For upgoing muon generation: Neutrino energy limits
  double MinEnu;
  double MaxEnu;
  double NuProdAlt;

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
  void setMultiMuon(bool MultiMu);
  void setMultiMuonFileName(std::string MultiMuonFileName);
  void setMultiMuonFileFirstEvent(int MultiMuFile1stEvt);
  void setMultiMuonNmin(int MultiMuNmin);
  void setTIFOnly_constant(bool TIF);
  void setTIFOnly_linear(bool TIF);
  void setMTCCHalf(bool MTCC);
  void setPlugVx(double PlugVtx);
  void setPlugVz(double PlugVtz);
  void setRhoAir(double VarRhoAir);
  void setRhoWall(double VarRhoSWall);
  void setRhoRock(double VarRhoRock);
  void setRhoClay(double VarRhoClay);
  void setRhoPlug(double VarRhoPlug);
  void setClayWidth(double ClayLaeyrWidth);

  void setMinEnu(double MinEn);
  void setMaxEnu(double MaxEn);
  void setNuProdAlt(double NuPrdAlt);
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
  // generate next multi muon event
  bool nextMultiEvent();
};
#endif
