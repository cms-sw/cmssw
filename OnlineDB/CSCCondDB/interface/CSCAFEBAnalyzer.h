#include <iostream>
#include <vector>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "OnlineDB/CSCCondDB/interface/CSCAFEBThrAnalysis.h"
#include "OnlineDB/CSCCondDB/interface/CSCAFEBConnectAnalysis.h"

class CSCAFEBAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit CSCAFEBAnalyzer(edm::ParameterSet const& conf);
  void analyze(edm::Event const& e, edm::EventSetup const& iSetup) override;
  void endJob() override;

private:
  /// variables persistent across events should be declared here.
  const std::string testname;
  const edm::InputTag CSCSrc_;
  const edm::EDGetTokenT<CSCWireDigiCollection> w_token;
  CSCAFEBThrAnalysis analysisthr_;
  CSCAFEBConnectAnalysis analysiscnt_;
};
