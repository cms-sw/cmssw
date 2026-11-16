#ifndef Integration_ThingExtSource_h
#define Integration_ThingExtSource_h

/** \class ThingExtSource
 *
 * \version   1st Version Dec. 27, 2005  

 * Comment added 3/17/2026:
 * As far as I can tell, the only purpose of this class is
 * to verify that when setRunAndEventInfo returns false
 * data processing stops. This source will request a stop
 * if the event number is greater than 2. OtherThingProducer
 * and OtherThingAnalyzer are run in the same test, but there
 * other unit tests that already do that. Failure mode is
 * that the unit test runs forever (probably times out
 * at some point). See inputExtSourceTest_cfg.py.
 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"
#include "ThingAlgorithm.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

namespace edmtest {
  class ThingExtSource : public edm::ProducerSourceBase {
  public:
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
