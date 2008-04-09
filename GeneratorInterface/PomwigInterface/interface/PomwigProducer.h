#ifndef PomwigProducer_h
#define PomwigProducer_h

/** \class PomwigProducer
 *
 * Generates Pomwig (Herwig) HepMC events
 *
 ***************************************/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <map>
#include <string>
#include "HepMC/GenEvent.h"

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
    double comenergy;
    std::string lhapdfSetPath_;
    bool useJimmy_;
    bool doMPInteraction_;
    bool printCards_;
    int numTrials_;

    double extCrossSect;
    double intCrossSect;
    double extFilterEff;

    double survivalProbability;
    int diffTopology;			
    bool enableForcedDecays;

    int maxEvents_;
    int eventNumber_;

    CLHEP::HepRandomEngine* fRandomEngine;
  };
} 

#endif
