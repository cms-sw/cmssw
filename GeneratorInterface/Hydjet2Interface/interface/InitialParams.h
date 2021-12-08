#ifndef INITALPARAMS
#define INITALPARAMS

struct InitialParamsHydjet_t {
  int fNevnt;     ///< number of events
  int femb;       ///< embedding flag
  double fSqrtS;  ///< cms energy per nucleon
  double fAw;     ///< atomic number of colliding nuclei
  int fIfb;       ///< \brief flag of type of centrality generation
                    ///< \details =0 is fixed by fBfix,
                    ///< not 0 impact parameter is generated in each event between fBfmin
                    ///< and fBmax according with Glauber model (f-la 30)
  double fBmin;   ///< minimum impact parameter in units of nuclear radius RA
  double fBmax;   ///< maximum impact parameter in units of nuclear radius RA
  double fBfix;   ///< fix impact parameter in units of nuclear radius RA

  int fSeed;  ///< \brief parameter to set the random nuber seed
                ///< \details =0 the current time is used to set the random generator seed,
                ///< !=0 the value fSeed is used to set the random generator seed
                ///< and then the state of random number generator in PYTHIA MRPY(1)=fSeed

  double fT;          ///< chemical freeze-out temperature in GeV
  double fMuB;        ///< baryon potential
  double fMuS;        ///< strangeness potential
  double fMuC;        ///< charm potential
  double fMuI3;       ///< isospin potential
  double fThFO;       ///< thermal freeze-out temperature T^th in GeV
  double fMu_th_pip;  ///< effective chemical potential of positivly charged pions at thermal in GeV

  double fTau;       ///< proper time value
  double fSigmaTau;  ///< its standart deviation (emission duration)
  double fR;         ///< maximal transverse radius
  double fYlmax;     ///< maximal longitudinal rapidity
  double fUmax;      ///< maximal transverse velocity multiplaed on gamma_r

  double frhou2;  ///< parameter to swich ON/OFF(0) rhou2
  double frhou3;  ///< parameter to swich ON/OFF(0) rhou3
  double frhou4;  ///< parameter to swich ON/OFF(0) rhou4

  double fDelta;    ///< momentum asymmetry parameter
  double fEpsilon;  ///< coordinate asymmetry parameter

  double fv2;  ///< parameter to swich ON/OFF(0) epsilon2 fluctuations
  double fv3;  ///< parameter to swich ON/OFF(0) epsilon3 fluctuations

  int fIfDeltaEpsilon;  ///< flag to specify fDelta and fEpsilon values(=0 user's ones, >=1 parametrized)

  int fDecay;         ///< flag to switch on/off hadron decays (=0 decays off, >=1 decays on), (default: 1)
  double fWeakDecay;  ///< flag to switch on/off weak hadron decays <0: decays off, >0: decays on, (default: 0)
  int fPythDecay;  ///< Flag to choose how to decay resonances in high-pt part, fPythDecay: 0 by PYTHIA decayer, 1 by FASTMC decayer(mstj(21)=0)

  int fEtaType;  ///< \brief flag to choose rapidity distribution
                   ///< \details if fEtaType<=0,
                   ///< then uniform rapidity distribution in [-fYlmax,fYlmax] if fEtaType>0,
                   ///< then Gaussian with dispertion = fYlmax

  int fTMuType;  ///< flag to use calculated chemical freeze-out temperature, baryon potential and strangeness potential as a function of fSqrtS

  double fCorrS;   ///< flag and value to include strangeness supression factor
  int fCharmProd;  ///< flag to include statistical charm production
  double fCorrC;   ///< flag and value to include charmness supression factor

  int fNhsel;  ///< \brief flag to switch on/off jet and hydro-state production
                 ///< \details 0: jet production off and hydro on,
                 ///< 1: jet production on and jet quenching off and hydro on,
                 ///< 2: jet production on and jet quenching on and hydro on,
                 ///< 3: jet production on and jet quenching off and hydro off,
                 ///<4: jet production on and jet quenching on and hydro off
  int fPyhist;  ///< Suppress PYTHIA particle history (=1 only final state particles from hard part; =0 include full particle history)
  int fIshad;  ///< \brief flag to switch on/off impact parameter dependent nuclear shadowing for gluons and light sea quarks (u,d,s)
      ///< \details 0: shadowing off,
      ///< 1: shadowing on for fAw=207, 197, 110, 40, default: 1

  double
      fPtmin;  ///< minimal transverse momentum transfer p_T of hard parton-parton scatterings in GeV (the PYTHIA parameter ckin(3)=fPtmin)

  //  PYQUEN energy loss model parameters:

  double fT0;  ///< \brief initial temperature (in GeV) of QGP for central Pb+Pb collisions at mid-rapidity
      ///< \details initial temperature for other centralities and atomic numbers will be calculated automatically
      ///< (allowed range is 0.2<fT0<2)

  double fTau0;  ///< proper QGP formation time in fm/c (0.01<fTau0<10)
  int fNf;       ///< number of active quark flavours N_f in QGP fNf=0, 1,2 or 3
  int fIenglu;   ///< \brief flag to fix type of in-medium partonic energy loss
                   ///< \details 0: radiative and collisional loss, 1: radiative loss only, 2:
                   ///< collisional loss only (default: 0);
  int fIanglu;   ///< \brief flag to fix type of angular distribution of in-medium emitted gluons
                   ///< \details 0: small-angular, 1: wide-angular, 2:collinear (default: 0).

  char partDat[256] = "";   ///< path to the particle data file
  char tabDecay[256] = "";  ///< path to the particle decay table

  bool fPythiaTune = false;   ///< Flag to use castom PYTHIA tune
  char pythiaTune[256] = "";  ///< path to the Pythia tune file

  bool doPrintInfo = true;       ///< Flag to turn ON/OFF additional info
  bool allowEmptyEvent = false;  ///< Allow or not empty events
};                                 ///< Structure of input parameters

#endif
