#ifndef Integration_OtherThingAnalyzer_h
#define Integration_OtherThingAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edmtest {

  class OtherThingAnalyzer : public edm::EDAnalyzer {
  public:

    explicit OtherThingAnalyzer(edm::ParameterSet const& pset);

    virtual ~OtherThingAnalyzer() {}

    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);

    void doit(edm::Event const& event, std::string const& label);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  private:
    bool thingWasDropped_;
    edm::InputTag otherTag_;
  };

}

#endif
