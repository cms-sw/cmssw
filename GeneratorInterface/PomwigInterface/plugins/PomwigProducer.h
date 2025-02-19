#ifndef PomwigProducer_h
#define PomwigProducer_h

/** \class PomwigProducer
 *
 * Generates Pomwig (Herwig) HepMC events
 *
 ***************************************/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HepMC/GenEvent.h"

#include <HepMC/IO_HERWIG.h>

class Run;
namespace CLHEP {
class HepRandomEngine;
}

namespace edm
{
  class PomwigProducer : public EDProducer {

  public:

    PomwigProducer(const ParameterSet &);
    virtual ~PomwigProducer();

    void endRun( Run& r);

  private:

    virtual void produce(Event & e, const EventSetup& es);
    void clear();

    bool hwgive(const std::string& iParm );
    bool setRngSeeds(int);

    HepMC::GenEvent  *evt;
    int herwigVerbosity_;
    bool herwigHepMCVerbosity_;
    int herwigLhapdfVerbosity_;
    int maxEventsToPrint_;
    double comEnergy_;
    bool useJimmy_;
    bool doMPInteraction_;
    bool printCards_;
    int numTrials_;

    double extCrossSect_;
    double extFilterEff_;

    double survivalProbability_;
    int diffTopology_;			
    int h1fit_;
    bool enableForcedDecays_;

    int eventNumber_;

    CLHEP::HepRandomEngine* fRandomEngine;

    HepMC::IO_HERWIG                conv;
  };
} 

#endif
