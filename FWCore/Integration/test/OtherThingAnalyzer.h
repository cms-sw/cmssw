#ifndef Integration_OtherThingAnalyzer_h
#define Integration_OtherThingAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edmreftest {

  class OtherThingAnalyzer : public edm::EDAnalyzer {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit OtherThingAnalyzer(edm::ParameterSet const&) {}

    virtual ~OtherThingAnalyzer() {}

    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);

  private:
  };

}

#endif
