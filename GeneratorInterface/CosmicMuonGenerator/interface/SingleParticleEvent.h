#ifndef SingleParticleEvent_h
#define SingleParticleEvent_h
//
// SingleParticleEvent by droll (23/DEC/2005)
//

// include files
#include "GeneratorInterface/CosmicMuonGenerator/interface/CosmicMuonParameters.h"
#include "GeneratorInterface/CosmicMuonGenerator/src/Point5MaterialMap.cc"
#include <iostream>

class SingleParticleEvent {
public:
  // constructor
  SingleParticleEvent() {
    ID = 0;
    Px = 0.;
    Py = 0.;
    Pz = 0.;
    E = 0.;
    M = 0.;
    Vx = 0.;
    Vy = 0.;
    Vz = 0.;
    T0 = 0.;
    ID_in = 0;
    Px_in = 0.;
    Py_in = 0.;
    Pz_in = 0.;
    E_in = 0.;
    M_in = 0.;
    Vx_in = 0.;
    Vy_in = 0.;
    Vz_in = 0.;
    T0_in = 0.;
    HitTarget = false;
    PlugVx = PlugOnShaftVx;
    PlugVz = PlugOnShaftVz;
    RhoAir = 0.;
    RhoWall = 0.;
    RhoRock = 0.;
    RhoClay = 0.;
    RhoPlug = 0.;
    ClayWidth = DefaultClayWidth;
  }
  // destructor
  ~SingleParticleEvent() {}

private:
  int ID;
  double Px;
  double Py;
  double Pz;
  double E;
  double M;
  double Vx;
  double Vy;
  double Vz;
  double T0;
  int ID_in;
  double Px_in;
  double Py_in;
  double Pz_in;
  double E_in;
  double M_in;
  double Vx_in;
  double Vy_in;
  double Vz_in;
  double T0_in;
  bool HitTarget;
  bool MTCC;

  // other stuff
  double dX;
  double dY;
  double dZ;
  double tmpVx;
  double tmpVy;
  double tmpVz;
  // update event during propagation
  void update(double stepSize);
  // temporary propagation
  void updateTmp(double stepSize);
  void subtractEloss(double waterEquivalents);  // update 4momentum
  double absVzTmp();                            // |Vz| [mm]
  double rVxyTmp();                             // R_XY [mm]

public:
  // create (initialize) an event with a single particle
  void create(int id, double px, double py, double pz, double e, double m, double vx, double vy, double vz, double t0);
  // propagate particle to target area
  void propagate(double ElossScaleFac,
                 double RadiusTarget,
                 double Z_DistTarget,
                 double Z_CentrTarget,
                 bool TrackerOnly,
                 bool MTCCHalf);
  double Eloss(double waterEquivalents, double Energy);  //return Eloss
  // particle has hit the target volume (during propagation)
  bool hitTarget();
  // event info (direct access)
  //initial state mother particle
  int id_in();                // [HEP particle code]
  double px_in();             // [GeV/c]
  double py_in();             // [GeV/c]
  double pz_in();             // [GeV/c]
  double e_in();              // [GeV]
  double m_in();              // [GeV/c^2]
  double vx_in();             // [mm]
  double vy_in();             // [mm]
  double vz_in();             // [mm]
  double t0_in();             // [mm/c] with c = 299.792458 mm/ns
  double WaterEquivalents();  //[g cm^-2]

  //final state daughter particles
  int id();     // [HEP particle code]
  double px();  // [GeV/c]
  double py();  // [GeV/c]
  double pz();  // [GeV/c]
  double e();   // [GeV]
  double m();   // [GeV/c^2]
  double vx();  // [mm]
  double vy();  // [mm]
  double vz();  // [mm]
  double t0();  // [mm/c] with c = 299.792458 mm/ns
  // event info (calculated)
  double phi();     // in horizontal (x-z) plane [rad]
  double theta();   // off vertical    (y) axis  [rad]
  double absmom();  // |p| [GeV/c]
  double absVz();   // |Vz| [mm]
  double rVxy();    // R_XY [mm]

  void setEug(double Eug);          // [GeV]
  double Eug();                     // [GeV/c]
  double deltaEmin(double Energy);  // [GeV]
  void SurfProj(double Vx_in,
                double Vy_in,
                double Vz_in,
                double Px_in,
                double Py_in,
                double Pz_in,
                double& Vx_up,
                double& Vy_up,
                double& Vz_up);

  double PlugVx;
  double PlugVz;
  double RhoAir;
  double RhoWall;
  double RhoRock;
  double RhoClay;
  double RhoPlug;
  double ClayWidth;
  double waterEquivalents;
  double E_ug;
};
#endif
