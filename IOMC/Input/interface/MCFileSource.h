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

namespace HepMC {
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
    ~MCFileSource() override;

  private:
    bool setRunAndEventInfo(EventID&, TimeValue_t& time, EventAuxiliary::ExperimentType& eType) override;
    void produce(Event& e) override;
    void clear();

    edm::propagate_const<HepMCFileReader*> reader_;
    edm::propagate_const<HepMC::GenEvent*> evt_;
    bool useExtendedAscii_;
  };
}  // namespace edm

#endif
