#ifndef Herwig6Source_h
#define Herwig6Source_h

/** \class Herwig6Source
 *
 * Generates Herwig HepMC events
 *
 ***************************************/


#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <map>
#include <string>
#include "CLHEP/HepMC/GenEvent.h"

namespace edm
{
  class Herwig6Source : public GeneratedInputSource {
  public:

    /// Constructor
    Herwig6Source(const ParameterSet &, const InputSourceDescription &);
    /// Destructor
    virtual ~Herwig6Source();


  private:

    /// Pass parameters to HERWIG
    bool hwgive(const std::string& iParm );

  private:
    
    virtual bool produce(Event & e);
    void clear();
    
    HepMC::GenEvent  *evt;
    
    /// Verbosity flag
    int herwigVerbosity_;
    bool herwigHepMCVerbosity_;
    int herwigLhapdfVerbosity_;
    int maxEventsToPrint_;
    double comenergy;

    std::string lhapdfSetPath_;
    
  };
} 

#endif
