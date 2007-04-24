#ifndef Pythia_Source_h
#define Pythia_Source_h

/** \class PythiaSource
 *
 * Generates Pythia HepMC events
 *
 * Hector Naves                                  
 *   for the Generator Interface. 26/10/05 
 * Patrick Janot
 *   read all possible cards for Pythia Setup. 26/02/06
 *   ( port from FAMOS )
 * Christian Veelken
 *   interface to TAUOLA tau decay library. 04/17/07
 *
 ***************************************/

#define PYCOMP pycomp_

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "GeneratorInterface/Pythia6Interface/interface/TauolaInterface.h"
#include <map>
#include <string>
#include "HepMC/GenEvent.h"
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RandFlat.h"

class Run;

namespace edm
{
  class PythiaSource : public GeneratedInputSource {
  public:

    /// Constructor
    PythiaSource(const ParameterSet &, const InputSourceDescription &);
    /// Destructor
    virtual ~PythiaSource();

    virtual void endRun( Run& r);

  private:

    /// Interface to the PYGIVE pythia routine, with add'l protections
    bool call_pygive(const std::string& iParm );
    bool call_txgive(const std::string& iParm );
    bool call_txgive_init();
    bool call_slhagive(const std::string& iParm );
    bool call_slha_init();
  
  private:
    
    virtual bool produce(Event & e);
    void clear();
    
    HepMC::GenEvent  *evt;
    
    /// Pythia PYLIST Verbosity flag
    unsigned int pythiaPylistVerbosity_;
    /// HepMC verbosity flag
    bool pythiaHepMCVerbosity_;
    /// Events to print if verbosity
    unsigned int maxEventsToPrint_;    
    
    // flag indicating whether or not TAUOLA is used for tau decays
    bool useTauola_;
    bool useTauolaPolarization_;

    // for single particle generation in pythia
    int    particleID;
    bool   doubleParticle;
    double ptmin, ptmax;
    double emin, emax;
    bool flatEnergy;
    double etamin, etamax;
    double phimin, phimax;
    double comenergy;
    
    CLHEP::HepRandomEngine* fRandomEngine;
    CLHEP::RandFlat*        fRandomGenerator; 

    TauolaInterface tauola_;

  };
} 

#endif
