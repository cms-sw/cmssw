#include <iostream>
#include <vector>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "OnlineDB/CSCCondDB/interface/CSCAFEBThrAnalysis.h"
#include "OnlineDB/CSCCondDB/interface/CSCAFEBConnectAnalysis.h"

class CSCAFEBAnalyzer : public edm::EDAnalyzer {
public:
  explicit CSCAFEBAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  virtual void endJob();
private:
  /// variables persistent across events should be declared here.
  std::string testname;
  CSCAFEBThrAnalysis analysisthr_;
  CSCAFEBConnectAnalysis analysiscnt_;

  edm::InputTag CSCSrc_;
};

