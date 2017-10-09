#ifndef IOMC_Input_MCFileSource_h
#define IOMC_Input_MCFileSource_h

/** \class MCFileSource
 *
 * Reads in HepMC events
 * Joanna Weng & Filip Moortgat 08/2005 
 ***************************************/

#include "FWCore/Sources/interface/ProducerSourceFromFiles.h"
#include "FWCore/Utilities/interface/propagate_const.h"

class HepMCFileReader;

namespace HepMC{
  class GenEvent;
}

namespace edm {
  class Event;
  class EventID;
  struct InputSourceDescription;
  class ParameterSet;
  class Timestamp;

  class MCFileSource : public ProducerSourceFromFiles {
  public:
    MCFileSource(const ParameterSet& pset, const InputSourceDescription& desc);
    virtual ~MCFileSource();

  private:
    virtual bool setRunAndEventInfo(EventID&, TimeValue_t& time, EventAuxiliary::ExperimentType& eType);
    virtual void produce(Event &e);
    void clear();
    
    edm::propagate_const<HepMCFileReader*> reader_;
    edm::propagate_const<HepMC::GenEvent*> evt_;
    bool useExtendedAscii_;
  };
} 

#endif
