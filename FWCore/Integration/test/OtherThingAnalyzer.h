#ifndef Integration_OtherThingAnalyzer_h
#define Integration_OtherThingAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edmtest {

  class OtherThingAnalyzer : public edm::EDAnalyzer {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit OtherThingAnalyzer(edm::ParameterSet const&) {}

    virtual ~OtherThingAnalyzer() {}

    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);

    virtual void beginLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& c);

    virtual void endLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& c);

    virtual void beginRun(edm::Run const& r, edm::EventSetup const& c);

    virtual void endRun(edm::Run const& r, edm::EventSetup const& c);

    void doit(edm::DataViewImpl const& dv, std::string const& label);

  private:
  };

}

#endif
