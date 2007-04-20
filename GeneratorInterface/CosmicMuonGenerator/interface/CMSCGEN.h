#ifndef CMSCGEN_h
#define CMSCGEN_h
//
// CMSCGEN.cc       T.Hebbeker 2006
//
// implemented in CMSSW by P. Biallass 29.03.2006  
//
//  original code in l3cgen.f from L3CGEN cosmic-generator:
/*
c
c simple l3 cosmic muon generator
c
c generates single muons at surface level in a region above the 
c L3 cosmic detector
c
c using a parametrisation of CORSIKA 5.20 results, obtained for primary
c   protons at L3 surface:
c energy range: 2 - 10000 GeV (using simple parametrisation of spectrum)
c zenith angle distribution 0-75 deg (|cos theta| = 1 ... 0.258)
c can be extrapolated up to theta=88 deg, but no guarantee for correctness!!!
c constant charge ratio 1.3 
c only muons from "above"
c
c accuracy of parametrisation of energy/angle/charge spectrum: 
c  about 5% in the energy range 10 GeV - 1000 GeV
*/
//
// speed on centrino 2 GHz inside root: 
//     1 million events (10 GeV, 0.5): 2 min
//
// reference: T. Hebbeker, A. Korn, L3+C note
//      "simulation programs for the L3 + Cosmics Experiment",
//      April 30, 1998 
//


#include "TMath.h" 
#include <iostream>
//#include "GeneratorInterface/CosmicMuonGenerator/interface/CosmicMuonGenerator.h"
#include "TRandom2.h"




class CMSCGEN 
{

// all units: GeV 

private:

  int initialization;  // energy and cos theta range set ?

  TRandom2 RanGen2; // random number generator (periodicity > 10**14)  

  float Emin;
  float Emax;
  float cmin;
  float cmax;

  float pq;
  float c; 

  float xemin;
  float xemax;

  float Emin_min;
  float Emin_max;

  float Emax_max;

  float cmin_min;
  float cmin_max;

  float cmin_allowed;

  float elmin;
  float elmax;
  float elfac;

  float c1;
  float c2;

  float cemax;

  float pe[9];
  float pc[3];
  float corr[101];

  bool TIFOnly_const;
  bool TIFOnly_lin;

public:

  // constructor
  CMSCGEN(){
  initialization = 0;
}

  //destructor
  ~CMSCGEN(){
  initialization = 0;
}

  int initialize(float,float,float,float,int,bool,bool);  
        // to set the energy and cos theta range 

  int generate();
       // to generate energy*charge and cos theta for one cosmic

  float energy_times_charge();

  float cos_theta();
   
};
#endif

