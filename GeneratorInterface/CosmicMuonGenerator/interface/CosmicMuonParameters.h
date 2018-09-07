#ifndef CosmicMuonParameters_h
#define CosmicMuonParameters_h
//
// Parameters for CosmicMuonGenerator by droll (05/DEC/2005)
//
//
// added plug and clay(moraine) specific constants, sonne (15/Jan/2009)
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
const double Z_PX56 = -14000.; // [mm] Z position of PX56 centre [mm]
/*
// densities of materials
const double RhoAir  = 0.001214; // [g cm^-3]
const double RhoWall = 2.5; // [g cm^-3]
const double RhoRock = 2.5; // [g cm^-3]
const double RhoClay = 2.3; // [g cm^-3]
const double RhoPlug = 2.5; // [g cm^-3]
*/
// width of clay layer between surface and rock
const double DefaultClayWidth = 50000.; // [mm]

//plug constants
const double PlugWidth = 2250.; // [mm]
const double PlugXlength = 20600.; // [mm]
const double PlugZlength = 16000.; // [mm]
const double PlugNoseXlength = 6400.; // [mm]
const double PlugNoseZlength = 1800.; // [mm]
const double PlugOnShaftVx = 0.; // [mm]
const double PlugOnShaftVz = Z_PX56; // [mm]

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



//define different materials
enum {Unknown=0, Plug, Wall, Air, Clay, Rock};


//Parameters for upward muons from neutrinos
const double N_A = 6.022e23; //mol^-1
const double alpha = 2.; //MeV/(g/cm^2)
const double beta_const = 3.9e-6; //(g/cm^2)^-1
const double epsilon = alpha/beta_const;
const double Rearth = 6370.e6; //mm

//Multi Muon relevant parameters
const double NorthCMSzDeltaPhi = 3./8.*Pi; //rad (Pi/2 if CMS -x = North)
const int max_Trials = 200000;


#endif
