#ifndef Streamer_ThingAnalyzer_h
#define Streamer_ThingAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <fstream>

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
    std::ofstream out_;
	int cnt_;
  };

}

#endif
