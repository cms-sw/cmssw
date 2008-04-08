#ifndef Alpgen_Producer_h
#define Alpgen_Producer_h

/** \class AlpgenProducer
 *
 * Generates Pythia HepMC events
 *
 * Hector Naves                                  
 *   for the Generator Interface. 26/10/05 
 * Patrick Janot
 *   read all possible cards for Pythia Setup. 26/02/06
 *   ( port from FAMOS )
 ***************************************/

#define PYCOMP pycomp_

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "HepMC/GenEvent.h"

namespace CLHEP
{
  class RandFlat ;
  class HepRandomEngine;
}

namespace edm
{
  class AlpgenProducer : public EDProducer {
  public:

    /// Constructor
    AlpgenProducer(const ParameterSet &);
    /// Destructor
    virtual ~AlpgenProducer();


  private:

    /// Interface to the PYGIVE/TXGIVE pythia routine, with add'l protections
    bool call_pygive(const std::string& iParm );
    bool call_txgive(const std::string& iParm );
    int Nev_; // number of events in the input file

    // .unw file with infos for AlpgenInfoProduct
    std::ifstream* unwfile;

  private:
    
    virtual void produce(Event & e, const EventSetup& es);
    void beginRun(Run& r);
    void clear();    
    HepMC::GenEvent  *evt;
    
    /// Pythia PYLIST Verbosity flag
    unsigned int pythiaPylistVerbosity_;
    /// HepMC verbosity flag
    bool pythiaHepMCVerbosity_;
    /// Events to print if verbosity
    unsigned int maxEventsToPrint_;    
    
    std::vector<std::string> fileNames_;
    std::string fileName_;
    unsigned int eventsRead_;
    
    // for single particle generation in pythia
    int    particleID;
    bool   doubleParticle;
    double ptmin, ptmax;
    double etamin, etamax;
    double phimin, phimax;
    
    CLHEP::HepRandomEngine* fRandomEngine;
    CLHEP::RandFlat*        fRandomGenerator; 
  };
} 

// 
#define alpgen_end alpgen_end_
    extern "C" {
        void alpgen_end();
      }

#endif
