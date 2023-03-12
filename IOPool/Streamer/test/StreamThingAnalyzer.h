#ifndef IOPool_Streamer_StreamThingAnalyzer_h
#define IOPool_Streamer_StreamThingAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
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

  class StreamThingAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit StreamThingAnalyzer(edm::ParameterSet const&);
    void endJob() override;

    void analyze(edm::Event const& e, edm::EventSetup const& c) override;

  private:
    std::string name_;
    int total_;
    std::ofstream out_;
    std::string inChecksumFile_;
    std::string outChecksumFile_;
    int cnt_;
    edm::GetterOfProducts<WriteThis> getterUsingLabel_;
  };
}  // namespace edmtest_thing

#endif
