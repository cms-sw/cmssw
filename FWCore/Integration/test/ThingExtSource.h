#ifndef Integration_ThingExtSource_h
#define Integration_ThingExtSource_h

/** \class ThingExtSource
 *
 * \version   1st Version Dec. 27, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/ProducerSourceFromFiles.h"
#include "FWCore/Integration/test/ThingAlgorithm.h"

namespace edmtest {
  class ThingExtSource : public edm::ProducerSourceFromFiles {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit ThingExtSource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc);

    ~ThingExtSource() override;

    bool setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&, edm::EventAuxiliary::ExperimentType&) override;

    void produce(edm::Event& e) override;

    void beginRun(edm::Run& r) override;

    void endRun(edm::Run& r) override;

    void beginLuminosityBlock(edm::LuminosityBlock& lb) override;

    void endLuminosityBlock(edm::LuminosityBlock& lb) override;

  private:
    ThingAlgorithm alg_;
  };
}
#endif
