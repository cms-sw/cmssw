#ifndef MCatNLOSource_h
#define MCatNLOSource_h

/** \class MCatNLOSource
 *
 * Generates MCatNLO HepMC events
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
  class MCatNLOSource : public GeneratedInputSource {
  public:

    /// Constructor
    MCatNLOSource(const ParameterSet &, const InputSourceDescription &);
    /// Destructor
    virtual ~MCatNLOSource();


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
    int maxEventsToPrint_;
    double comenergy;
    int processNumber_;
  };
} 

#endif
