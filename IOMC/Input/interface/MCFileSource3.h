#ifndef IOMC_Input_MCFileSource3_h
#define IOMC_Input_MCFileSource3_h

/** \class MCFileSource3
 *
 * Reads in HepMC3 events, the HepMC3 counterpart of MCFileSource
 ***************************************/

#include <memory>

#include "FWCore/Sources/interface/ProducerSourceBase.h"
#include "FWStorage/Catalog/interface/InputFileCatalog.h"

class HepMC3FileReader;

namespace HepMC3 {
  class GenEvent;
}

namespace edm {
  class ConfigurationDescriptions;
  class Event;
  class EventID;
  struct InputSourceDescription;
  class ParameterSet;

  class MCFileSource3 : public ProducerSourceBase {
  public:
    MCFileSource3(const ParameterSet& pset, const InputSourceDescription& desc);
    ~MCFileSource3() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    bool setRunAndEventInfo(EventID&, TimeValue_t& time, EventAuxiliary::ExperimentType& eType) override;
    void produce(Event& e) override;

    std::unique_ptr<HepMC3FileReader> reader_;
    const HepMC3::GenEvent* evt_;
    InputFileCatalog inputFileCatalog_;
    const bool printEvent_;
  };
}  // namespace edm

#endif
