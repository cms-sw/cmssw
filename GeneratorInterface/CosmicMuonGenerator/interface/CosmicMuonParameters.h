#ifndef CosmicMuonParameters_h
#define CosmicMuonParameters_h
//
// Parameters for CosmicMuonGenerator by droll (05/DEC/2005)
//
#include "TMath.h"

// flags
const bool Debug = false; // debugging printout
const bool EventDisplay = true; // display single events (if ROOT_INTERACTIVE is defined as true)

// algorithmic constants
const double MinStepSize = 10.; // minimal propagation step size [mm] must be small compared to target size
// mathematical constants
const double Pi = acos(-1.); // [rad]
const double TwoPi = 2.0*Pi; // [rad]
const double Deg2Rad = Pi/180.; // [deg] -> [rad]
const double Rad2Deg = 180./Pi; // [rad] -> [deg]
// physical constants
const double SpeedOfLight = 299.792458; // [mm/ns]
const double MuonMass = 0.105658357; // [GeV/c^2]
//const double ChargeFrac = 0.545454545; // n(mu+)/n(mu-) ~ 1.2 defined in CMSCGEN
// geometry
const double SurfaceOfEarth = 88874.; // Y-distance to surface of earth [mm]
const double Z_PX56 = 14000.; // Z-distance to central axis of PX 56 [mm]
// densities of materials
const double RhoAir  = 0.00; // [g cm^-3]
const double RhoWall = 2.65; // [g cm^-3]
const double RhoRock = 2.50; // [g cm^-3]
// cylinder around CMS (with R, +-Z)
// WARNING: These values will be set to tracker-only setup if "TrackerOnly=true" in .cfg-file. 
// This means R=1200 and Z=2800, no material or B-field outside is considered
const double RadiusCMS =  8000.; // [mm]
const double Z_DistCMS = 15000.; // [mm]
const double RadiusTracker =  1200.; // [mm]
const double Z_DistTracker = 2800.; // [mm]
// cylinder actually used in the code
//const double RadiusTarget = RadiusCMS; // [mm]  //now controlled by cfg-file!!!
//const double Z_DistTarget = Z_DistCMS; // [mm]  //now controlled by cfg-file!!!

#endif
