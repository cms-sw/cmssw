#ifndef SingleParticleEvent_h
#define SingleParticleEvent_h
//
// SingleParticleEvent by droll (23/DEC/2005)
//

// include files
#include "GeneratorInterface/CosmicMuonGenerator/interface/CosmicMuonParameters.h"
#include "GeneratorInterface/CosmicMuonGenerator/src/Point5MaterialMap.cc"
#include <iostream>


class SingleParticleEvent{
public:
  // constructor
  SingleParticleEvent(){
    ID = 0;
    Px = 0.; Py = 0.; Pz = 0.; E = 0.; M = 0.;
    Vx = 0.; Vy = 0.; Vz = 0.; T0 = 0.;
    HitTarget = false;
    PlugVx = PlugOnShaftVx;
    PlugVz = PlugOnShaftVz;
  }
  // destructor
  ~SingleParticleEvent(){}
private:
  int ID;
  double Px; double Py; double Pz; double E; double M;
  double Vx; double Vy; double Vz; double T0;
  bool HitTarget;
  bool MTCC;

  // other stuff
  double dX; double dY; double dZ;
  double tmpVx; double tmpVy; double tmpVz;
  // update event during propagation
  void update(double stepSize);
  // temporary propagation
  void updateTmp(double stepSize);
  void subtractEloss(double waterEquivalents); // update 4momentum
  double absVzTmp(); // |Vz| [mm]
  double rVxyTmp();  // R_XY [mm]

public:
  // create (initialize) an event with a single particle
  void create(int id, double px, double py, double pz, double e, double m, double vx, double vy, double vz, double t0);
  // propagate particle to target area
  void propagate(double ElossScaleFac, double RadiusTarget, double Z_DistTarget, bool TrackerOnly, bool MTCCHalf);

  // particle has hit the target volume (during propagation)
  bool hitTarget();
  // event info (direct access)
  int    id(); // [HEP particle code]
  double px(); // [GeV/c]
  double py(); // [GeV/c]
  double pz(); // [GeV/c]
  double e();  // [GeV]
  double m();  // [GeV/c^2]
  double vx(); // [mm]
  double vy(); // [mm]
  double vz(); // [mm]
  double t0(); // [mm/c] with c = 299.792458 mm/ns
  // event info (calculated)
  double phi();   // in horizontal (x-z) plane [rad]
  double theta(); // off vertical    (y) axis  [rad]
  double absmom(); // |p| [GeV/c]
  double absVz(); // |Vz| [mm]
  double rVxy();  // R_XY [mm]

  double PlugVx;
  double PlugVz;
};
#endif
