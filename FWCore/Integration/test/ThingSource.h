#ifndef Integration_ThingSource_h
#define Integration_ThingSource_h

/** \class ThingSource
 *
 * \version   1st Version Dec. 27, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Integration/test/ThingAlgorithm.h"

namespace edmtest {
  class ThingSource : public edm::GeneratedInputSource {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit ThingSource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc);

    virtual ~ThingSource();

    virtual bool produce(edm::Event& e);

    virtual void beginRun(edm::Run& r);

    virtual void endRun(edm::Run& r);

    virtual void beginLuminosityBlock(edm::LuminosityBlock& lb);

    virtual void endLuminosityBlock(edm::LuminosityBlock& lb);

  private:
    ThingAlgorithm alg_;
  };
}
#endif
