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
    
    MCatNLOSource(const ParameterSet &, const InputSourceDescription &);
    virtual ~MCatNLOSource();
    
  private:
    
    virtual bool produce(Event & e);
    void clear();
    bool give(const std::string& iParm );
    bool hwgive(const std::string& iParm );
    void processHG();
    void processLL();
    void processVH();
    void processVV();
    void processQQ();
    void processSB();
    void processST();
    void processUnknown(bool);
    void getVpar();

    void createStringFile(const std::string&);
    
    HepMC::GenEvent  *evt;

    // include hard event generation ... 
    bool doHardEvents_;
    
    // Verbosity parameters
    int mcatnloVerbosity_;
    int herwigVerbosity_;
    bool herwigHepMCVerbosity_;
    int maxEventsToPrint_;

    // run parameters
    double comenergy;
    int processNumber_;
    int numEvents_;
    std::string stringFileName_;

    // needed for NLO ouput
    char directory[70];
    char prefix_bases[10];
    char prefix_events[10];

    std::string lhapdfSetPath_;
    bool useJimmy_;
    bool doMPInteraction_;
    int numTrials_;
    bool printCards_;

  };
} 

#endif
