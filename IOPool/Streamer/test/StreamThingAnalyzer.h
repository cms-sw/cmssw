#ifndef IOPool_Streamer_StreamThingAnalyzer_h
#define IOPool_Streamer_StreamThingAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"

#if 1
#include "DataFormats/TestObjects/interface/StreamTestThing.h"
typedef edmtestprod::StreamTestThing WriteThis;
#else
#include "FWCore/Integration/interface/IntArray.h"
typedef edmtestprod::IntArray WriteThis;
#endif

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
    edm::GetterOfProducts<WriteThis> getterUsingLabel_;
  };
}

#endif
