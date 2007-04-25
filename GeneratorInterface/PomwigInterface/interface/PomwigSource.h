#ifndef PomwigSource_h
#define PomwigSource_h

/** \class PomwigSource
 *
 * Generates Pomwig HepMC events
 *
 ***************************************/


#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <map>
#include <string>
#include "HepMC/GenEvent.h"

namespace edm
{
  class PomwigSource : public GeneratedInputSource {

  public:

    PomwigSource(const ParameterSet &, const InputSourceDescription &);
    virtual ~PomwigSource();


  private:

    virtual bool produce(Event & e);
    void clear();

    bool hwgive(const std::string& iParm );
    bool setRngSeeds(int);

    HepMC::GenEvent  *evt;
    int herwigVerbosity_;
    bool herwigHepMCVerbosity_;
    int herwigLhapdfVerbosity_;
    int maxEventsToPrint_;
    double comenergy;
    std::string lhapdfSetPath_;
    bool useJimmy_;
    bool doMPInteraction_;
    bool printCards_;
    int numTrials_;

    int diffTopology;	
  };
} 

#endif
