#ifndef Integration_ThingSource_h
#define Integration_ThingSource_h

/** \class ThingSource
 *
 * \version   1st Version Dec. 27, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Integration/test/ThingAlgorithm.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"

namespace edmtest {
  class ThingSource : public edm::ProducerSourceBase {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit ThingSource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc);

    ~ThingSource() override;

    bool setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&, edm::EventAuxiliary::ExperimentType&) override {return true;}

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
