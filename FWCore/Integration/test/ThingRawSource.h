#ifndef Integration_ThingRawSource_h
#define Integration_ThingRawSource_h

/** \class ThingRawSource
 *
 * \version   1st Version Dec. 27, 2005  

 *
 ************************************************************/

#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/RawInputSource.h"
#include "FWCore/Integration/test/ThingAlgorithm.h"

namespace edmtest {
  class ThingRawSource : public edm::RawInputSource {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit ThingRawSource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc);

    virtual ~ThingRawSource();

    virtual std::auto_ptr<edm::Event> readOneEvent();

    virtual void beginRun(edm::Run& r);

    virtual void endRun(edm::Run& r);

    virtual void beginLuminosityBlock(edm::LuminosityBlock& lb);

    virtual void endLuminosityBlock(edm::LuminosityBlock& lb);

  private:
    ThingAlgorithm alg_;
    edm::EventID eventID_;
  };
}
#endif
