#ifndef CMSCGEN_h
#define CMSCGEN_h

//
// CMSCGEN.cc  version 3.0     Thomas Hebbeker 2007-05-15  
//
// implemented in CMSSW by P. Biallass 2007-05-28 
//
// documentation: CMS internal note 2007 "Improved Parametrization of the Cosmic Muon Flux for the generator CMSCGEN" by Biallass + Hebbeker
//
// inspired by fortran version l3cgen.f, version 4.0, 1999
//
// history: new version of parametrization of energy and angular distribution of cosmic muons,
//          now based on new version 6.60 of CORSIKA (2007). Revisited parametrization, now using slightly different polynomials and new coefficients.
//
// new range: 3...3000 GeV, cos(incident angle) = 0.1...1 which means theta=0°...84.26° where z-axis vertical axis  
//            Now parametrization obtained from full range, thus no extrapolation to any angles or energies needed any more.
// accuracy: now well known, see internal note for details
//           7% for range 10...500 GeV, 50% for 3000 GeV and 25% for 3 GeV



#include "TMath.h" 
#include <iostream>
//#include "GeneratorInterface/CosmicMuonGenerator/interface/CosmicMuonGenerator.h"
#include "TRandom2.h"


class CMSCGEN 
{

// all units: GeV 

private:

  int initialization;  // energy and cos theta range set ?

  double pmin;
  double pmax;
  double cmin;
  double cmax;
  double cmin_in;
  double cmax_in;

  double pq;
  double c; 

  double xemin;
  double xemax;

  double pmin_min;
  double pmin_max;

  double cmax_min;
  double cmax_max;

  double Lmin;
  double Lmax;
  double Lfac;

  double c1;
  double c2;

  double b0;
  double b1;
  double b2;

  double integrated_flux;

  double cemax;

  double pe[9];
  double b0c[3], b1c[3], b2c[3];
  double corr[101];


  TRandom2 RanGen2; // random number generator (periodicity > 10**14)  


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

  int initialize(double,double,double,double,int,bool,bool);  
        // to set the energy and cos theta range 

  int generate();
       // to generate energy*charge and cos theta for one cosmic

  double momentum_times_charge();

  double cos_theta();

  double flux();
   
};
#endif

