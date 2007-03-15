#ifndef PomwigSource_h
#define PomwigSource_h

/** \class PomwigSource 
 *
 * Generates Pomwig HepMC events
 *
 * Based on Herwig6Source
 ***************************************/


#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <map>
#include <string>
#include "CLHEP/HepMC/GenEvent.h"

namespace edm
{
  class PomwigSource : public GeneratedInputSource {
  public:

    /// Constructor
    PomwigSource(const ParameterSet &, const InputSourceDescription &);
    /// Destructor
    virtual ~PomwigSource();


  private:

    virtual bool produce(Event & e);
    void clear();
    
    bool hwgive(const std::string& iParm );	
    bool setRngSeeds(int);
	
    HepMC::GenEvent  *evt;
    
    /// Verbosity flag
    int herwigVerbosity_;
    bool herwigHepMCVerbosity_;
    int herwigLhapdfVerbosity_;
    int maxEventsToPrint_;
    double comenergy;
    int diffTopology;	

    std::string lhapdfSetPath_;
    bool printCards_;
  };
} 

#endif
