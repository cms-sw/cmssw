#ifndef Exhume_Producer_h
#define Exhume_Producer_h

/** \class ExhumeProducer
 *
 * Generates ExHuME (Pythia for hadronization) HepMC events
 *
 * Based on PythiaProducer
 ***************************************/

#define PYCOMP pycomp_

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <map>
#include <string>
#include "HepMC/GenEvent.h"
//#include "CLHEP/Random/JamesRandom.h"
//#include "CLHEP/Random/RandFlat.h"

//ExHuME headers
#include "GeneratorInterface/ExhumeInterface/interface/Event.h"
#include "GeneratorInterface/ExhumeInterface/interface/QQ.h"
#include "GeneratorInterface/ExhumeInterface/interface/GG.h"
#include "GeneratorInterface/ExhumeInterface/interface/Higgs.h"

#include "GeneratorInterface/ExhumeInterface/interface/PYR.h"

class Run;
namespace CLHEP {
class HepRandomEngine;
class RandFlat;
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
    
    double comenergy;
	
    // external cross section and filter efficiency
    double extCrossSect;
    double extFilterEff;
	
    CLHEP::HepRandomEngine* fRandomEngine;
    CLHEP::RandFlat*        fRandomGenerator;

    Exhume::Event* ExhumeEvent;	
    //Exhume::Higgs* higgs;
    Exhume::CrossSection* ExhumeProcess;

    std::string ProcessType;	
    int HiggsDecay;       //for Higgs
    int QuarkType;        //for QQ	
    double ThetaMin;	  //for QQ and GG
    double MassRangeLow;
    double MassRangeHigh;

    int sigID;		

// Added by JMM
    unsigned int eventNumber_;
// End of JMM insertion
  };
} 

#endif
