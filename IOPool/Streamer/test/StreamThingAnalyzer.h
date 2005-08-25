#ifndef IOP_THINGANALYZER_H
#define IOP_THINGANALYZER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

namespace edmtest_thing {

  class StreamThingAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit StreamThingAnalyzer(edm::ParameterSet const&);
    
    virtual ~StreamThingAnalyzer();

    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);

  private:
    std::string name_;
    int total_;
  };

}

#endif
