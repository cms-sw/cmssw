#ifndef Herwig6Producer_h
#define Herwig6Producer_h

/** \class Herwig6Producer
 *
 * Generates Herwig HepMC events
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
  class Herwig6Producer : public EDProducer {

  public:

    Herwig6Producer(const ParameterSet &);
    virtual ~Herwig6Producer();

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
 
    unsigned int numberEvents_;

    CLHEP::HepRandomEngine*	fRandomEngine;

  };
} 

#endif
