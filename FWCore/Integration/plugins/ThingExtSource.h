#ifndef Integration_ThingExtSource_h
#define Integration_ThingExtSource_h

/** \class ThingExtSource
 *
 * \version   1st Version Dec. 27, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/ProducerSourceFromFiles.h"
#include "ThingAlgorithm.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

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

    void beginLuminosityBlock(edm::LuminosityBlock& lb) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    //Not called by the framework, only used internally
    void endRun(edm::Run& r);
    void endLuminosityBlock(edm::LuminosityBlock& lb);

    ThingAlgorithm alg_;
  };
}  // namespace edmtest
#endif
