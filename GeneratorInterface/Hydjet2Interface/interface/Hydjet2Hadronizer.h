/*                                                                            
 Based on class InitalStateHydjet:                                          
 Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
 amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
 November. 2, 2005                                

*/

#ifndef Hydjet2Hadronizer_h
#define Hydjet2Hadronizer_h

/** \class Hydjet2Hadronizer
 *
 * Generates HYDJET++ ==> HepMC events
 *
 * Andrey Belyaev
 * for the Generator Interface. Sep 2014
 *********************************************/

#include "DatabasePDG.h"
#include "Particle.h"
#include "InitialState.h"

#define PYCOMP pycomp_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoisson.h"
#include "CLHEP/Random/RandGauss.h"

#include <map>
#include <string>
#include <vector>
#include <math.h>

#include "HYJET_COMMONS.h"
extern HYIPARCommon HYIPAR;
extern HYFPARCommon HYFPAR;
extern HYJPARCommon HYJPAR;
extern HYPARTCommon HYPART;
extern SERVICECommon SERVICE;

#define kMax 500000

namespace CLHEP {
  class HepRandomEngine;
  class RandFlat;
  class RandPoisson;
  class RandGauss;
}

CLHEP::HepRandomEngine* hjRandomEngine;

namespace HepMC {
  class GenEvent;
  class GenParticle;
  class GenVertex;
}

namespace gen
{ 
  class Pythia6Service;
  class Hydjet2Hadronizer : public InitialState,  public BaseHadronizer {


  public:

    Hydjet2Hadronizer(const edm::ParameterSet&);
    ~Hydjet2Hadronizer();

    bool readSettings( int );
    bool declareSpecialSettings( const std::vector<std::string>& ) { return true; }
    bool initializeForInternalPartons();
    bool initializeForExternalPartons();//0
    bool generatePartonsAndHadronize();
    bool declareStableParticles( const std::vector<int>& );

    bool hadronize();//0
    bool decay();//0
    bool residualDecay();
    void finalizeEvent();
    void statistics();
    const char* classname() const;

    //________________________________________
  
    void SetVolEff(double value) {fVolEff = value;}
    double GetVolEff() {return fVolEff;}
    virtual bool RunDecays() {return (fDecay>0 ? kTRUE : kFALSE);}
    virtual double GetWeakDecayLimit() {return fWeakDecay;}  

    bool IniOfThFreezeoutParameters();
 
    double f(double);
    double f2(double, double, double);
  
    double SimpsonIntegrator(double, double, double, double);
    double SimpsonIntegrator2(double, double, double, double);
    double MidpointIntegrator2(double, double, double, double);
    double CharmEnhancementFactor(double, double, double, double);

  private:

    virtual void doSetRandomEngine(CLHEP::HepRandomEngine* v) override;
    void rotateEvtPlane();
    bool	get_particles(HepMC::GenEvent* evt);
    HepMC::GenParticle*	build_hyjet2( int index, int barcode );
    HepMC::GenVertex* build_hyjet2_vertex(int i, int id);
    void	add_heavy_ion_rec(HepMC::GenEvent *evt);
   
    virtual std::vector<std::string> const& doSharedResources() const override { return theSharedResources; }
    static const std::vector<std::string> theSharedResources;

    inline double nuclear_radius() const;

    double fVolEff;                           // the effective volume

    // the list of initial state parameters
 
    //int fNevnt;                      // number of events
    double fSqrtS;                   // cms energy per nucleon
    double fAw;                      // atomic number of colliding nuclei
    int fIfb;                        // flag of type of centrality generation (=0 is fixed by fBfix, not 0 
                                     // impact parameter is generated in each event between fBfmin 
                                     // and fBmax according with Glauber model (f-la 30)
    double fBmin;                    // minimum impact parameter in units of nuclear radius RA 
    double fBmax;                    // maximum impact parameter in units of nuclear radius RA
    double fBfix;                    // fix impact parameter in units of nuclear radius RA
         
    double fT;                       // chemical freeze-out temperature in GeV    
    double fMuB;                     // baryon potential 
    double fMuS;                     // strangeness potential 
    double fMuC;                     // charm potential 
    double fMuI3;                    // isospin potential   
    double fThFO;                    // thermal freeze-out temperature T^th in GeV
    double fMu_th_pip;               // effective chemical potential of positivly charged pions at thermal in GeV 
       
    double fTau;                     // proper time value
    double fSigmaTau;                // its standart deviation (emission duration)
    double fR;                       // maximal transverse radius 
    double fYlmax;                   // maximal longitudinal rapidity 
    double fUmax;                    // maximal transverse velocity multiplaed on \gamma_r 
    double fDelta;                   // momentum asymmetry parameter
    double fEpsilon;                 // coordinate asymmetry parameter
    int    fIfDeltaEpsilon;          // flag to specify fDelta and fEpsilon values(=0 user's ones, >=1 parametrized)
  
    int fDecay;                      // flag to switch on/off hadron decays (=0 decays off, >=1 decays on), (default: 1)
    double fWeakDecay;               // flag to switch on/off weak hadron decays <0: decays off, >0: decays on, (default: 0)
    int fPythDecay;                  // Flag to choose how to decay resonances in high-pt part, fPythDecay: 0 by PYTHIA decayer, 
                                     // 1 by FASTMC decayer(mstj(21)=0)  
  
    int fEtaType;                    // flag to choose rapidity distribution, if fEtaType<=0, 
                                     // then uniform rapidity distribution in [-fYlmax,fYlmax] if fEtaType>0,
                                     // then Gaussian with dispertion = fYlmax 
  
    int fTMuType;                    // flag to use calculated chemical freeze-out temperature,
                                     // baryon potential and strangeness potential as a function of fSqrtS 

    double fCorrS;                   // flag and value to include strangeness supression factor    
    int fCharmProd;                  // flag to include statistical charm production    
    double fCorrC;                   // flag and value to include charmness supression factor

    int fNhsel;                      // flag to switch on/off jet and hydro-state production (0: jet
                                     // production off and hydro on, 1: jet production on and jet quenching
                                     // off and hydro on, 2: jet production on and jet quenching on and
                                     // hydro on, 3: jet production on and jet quenching off and hydro
                                     // off, 4: jet production on and jet quenching on and hydro off
    int fPyhist;                     // Suppress PYTHIA particle history (=1 only final state particles from hard part; =0 include full particle history) 
    int fIshad;                      // flag to switch on/off impact parameter dependent nuclear
                                     // shadowing for gluons and light sea quarks (u,d,s) (0: shadowing off,
                                     // 1: shadowing on for fAw=207, 197, 110, 40, default: 1
  
    double fPtmin;                   // minimal transverse momentum transfer p_T of hard
                                     // parton-parton scatterings in GeV (the PYTHIA parameter ckin(3)=fPtmin)
   
    //  PYQUEN energy loss model parameters:
 
    double fT0;                      // initial temperature (in GeV) of QGP for
                                     // central Pb+Pb collisions at mid-rapidity (initial temperature for other
                                     // centralities and atomic numbers will be calculated automatically) (allowed range is 0.2<fT0<2) 
  
    double fTau0;                    // proper QGP formation time in fm/c (0.01<fTau0<10)
    int fNf;                         // number of active quark flavours N_f in QGP fNf=0, 1,2 or 3 
    int fIenglu;                     // flag to fix type of in-medium partonic energy loss 
                                     // (0: radiative and collisional loss, 1: radiative loss only, 2:
                                     // collisional loss only) (default: 0);
    int fIanglu;                     // flag to fix type of angular distribution of in-medium emitted
                                     // gluons (0: small-angular, 1: wide-angular, 2:collinear) (default: 0).



    bool embedding_; // Switch for embedding mode
    bool rotate_; // Switch to rotate event plane
    HepMC::GenEvent *evt;
    int nsub_; // number of sub-events
    int nhard_; // multiplicity of PYTHIA(+PYQUEN)-induced particles in event
    int nsoft_; // multiplicity of HYDRO-induced particles in event 
    double phi0_; // Event plane angle
    double sinphi0_;
    double cosphi0_;
    Pythia6Service* pythia6Service_;

    unsigned int pythiaPylistVerbosity_; // pythia verbosity; def=1 
    unsigned int maxEventsToPrint_; // Events to print if verbosity 


    edm::InputTag src_;
  
    int    fNPartTypes;              //counter of hadron species  
    int    fPartEnc[1000];           //Hadron encodings. Maximal number of hadron species is 1000!!!
    double fPartMult[2000];          //Multiplicities of hadron species
    double fPartMu[2000];            //Chemical potentials of hadron species
    double fMuTh[1000];              //Chemical potentials at thermal freezeout of hadron species

    //open charm  
    double fNocth;
    double fNccth;
 
    edm::ParameterSet pset;
    edm::Service<TFileService> fs;

    int ev, sseed, Njet, Nbcol, Npart, Ntot, Npyt, Nhyd;
    float Bgen, Sigin, Sigjet;
    float Px[kMax];
    float Py[kMax];
    float Pz[kMax];
    float E[kMax];
    float X[kMax];
    float Y[kMax];
    float Z[kMax];
    float T[kMax];
    int pdg[kMax];
    int Mpdg[kMax];
    int type[kMax];
    int pythiaStatus[kMax];
    int Index[kMax];
    int MotherIndex[kMax];
    int NDaughters[kMax];
    int FirstDaughterIndex[kMax];
    int LastDaughterIndex[kMax];
    int final[kMax];

    ParticleAllocator allocator;
    List_t source;

  }; 
  double Hydjet2Hadronizer::nuclear_radius() const
  {
    // Return the nuclear radius derived from the
    // beam/target atomic mass number.
    return 1.15 * pow((double)fAw, 1./3.);
  }
} /*end namespace*/
#endif
