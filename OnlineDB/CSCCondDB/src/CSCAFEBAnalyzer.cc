#include <iostream>
#include <vector>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "OnlineDB/CSCCondDB/interface/CSCAFEBAnalyzer.h"

CSCAFEBAnalyzer::CSCAFEBAnalyzer(edm::ParameterSet const& conf) {
  /// If your module takes parameters, here is where you would define
  /// their names and types, and access them to initialize internal
  /// variables. Example as follows:

  testname = conf.getParameter<std::string>("TestName");

  if (testname == "AFEBThresholdScan")
    analysisthr_.setup(conf.getParameter<std::string>("HistogramFile"));
  if (testname == "AFEBConnectivity")
    analysiscnt_.setup(conf.getParameter<std::string>("HistogramFile"));

  /// get labels for input tags

  CSCSrc_ = conf.getParameter<edm::InputTag>("CSCSrc");
}

void CSCAFEBAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  edm::Handle<CSCWireDigiCollection> wire_digis;

  /// For CSC unpacker

  //   const char* modtag="cscunpacker";
  //   e.getByLabel(modtag,"MuonCSCWireDigi",wire_digis);
  e.getByLabel(CSCSrc_, wire_digis);

  if (testname == "AFEBThresholdScan")
    analysisthr_.analyze(*wire_digis);
  if (testname == "AFEBConnectivity")
    analysiscnt_.analyze(*wire_digis);
}

void CSCAFEBAnalyzer::endJob() {
  if (testname == "AFEBThresholdScan")
    analysisthr_.done();
  if (testname == "AFEBConnectivity")
    analysiscnt_.done();
}
