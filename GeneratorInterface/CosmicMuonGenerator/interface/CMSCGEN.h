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

#include <iostream>

#include "GeneratorInterface/CosmicMuonGenerator/interface/CosmicMuonParameters.h"

namespace CLHEP {
  class HepRandomEngine;
}

class CMSCGEN {
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

  CLHEP::HepRandomEngine* RanGen2;  // random number generator
  bool delRanGen;

  bool TIFOnly_const;
  bool TIFOnly_lin;

  //variables for upgoing muons from neutrinos
  double enumin;
  double enumax;

public:
  // constructor
  CMSCGEN();

  //destructor
  ~CMSCGEN();

  void setRandomEngine(CLHEP::HepRandomEngine* v);

  // to set the energy and cos theta range
  int initialize(double, double, double, double, CLHEP::HepRandomEngine*, bool, bool);
  int initialize(double, double, double, double, int, bool, bool);

  int generate();
  // to generate energy*charge and cos theta for one cosmic

  double momentum_times_charge();

  double cos_theta();

  double flux();

  //upward going muons from neutrinos
  int initializeNuMu(double, double, double, double, double, double, double, double, double, CLHEP::HepRandomEngine*);
  int initializeNuMu(double, double, double, double, double, double, double, double, double, int);
  int generateNuMu();

  double Rnunubar;  //Ration of nu to nubar
  double ProdAlt;   //production altitude in atmosphere
  double sigma;
  double AR;
  double dNdEmudEnu(double Enu, double Emu, double theta);
  double dNdEmudEnuMax;
  double negabs, negfrac;
};
#endif
