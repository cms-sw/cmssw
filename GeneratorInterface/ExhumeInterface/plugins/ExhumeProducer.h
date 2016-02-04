#ifndef Exhume_Producer_h
#define Exhume_Producer_h

/** \class ExhumeProducer
 *
 * Generates ExHuME (Pythia for hadronization) HepMC events
 *
 * Based on PythiaProducer
 ***************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HepMC/GenEvent.h"

class Run;
namespace CLHEP {
class HepRandomEngine;
class RandFlat;
}

namespace Exhume {
class Event;
class CrossSection;
}

namespace edm
{
  class ExhumeProducer : public EDProducer {

  public:

    /// Constructor
    ExhumeProducer(const ParameterSet &);
    /// Destructor
    virtual ~ExhumeProducer();

    void endRun( Run& r);	

  private:

    /// Interface to the PYGIVE pythia routine, with add'l protections
    //bool call_pygive(const std::string& iParm );
    //bool call_txgive(const std::string& iParm );
    //bool call_txgive_init();
  
  private:
    
    virtual void produce(edm::Event & e, const EventSetup& es);
    void clear();
    
    HepMC::GenEvent  *evt;
    
    /// Pythia PYLIST Verbosity flag
    unsigned int pythiaPylistVerbosity_;
    /// HepMC verbosity flag
    bool pythiaHepMCVerbosity_;
    /// Events to print if verbosity
    unsigned int maxEventsToPrint_;    
    
    double comEnergy_;
	
    // external cross section and filter efficiency
    double extCrossSect_;
    double extFilterEff_;
	
    CLHEP::HepRandomEngine* fRandomEngine;
    CLHEP::RandFlat*        fRandomGenerator;

    Exhume::Event* exhumeEvent_;	
    Exhume::CrossSection* exhumeProcess_;

    int sigID_;		

// Added by JMM
    unsigned int eventNumber_;
// End of JMM insertion
  };
} 

#endif
