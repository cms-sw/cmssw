#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <cmath>
#include <iostream>

namespace edm {
  class EventSetup;
}

using namespace std;
using namespace edm;

namespace edmtest {

  class TestExpressLumiProducer : public edm::one::EDAnalyzer<edm::one::WatchLuminosityBlocks> {
  public:
    explicit TestExpressLumiProducer(edm::ParameterSet const&);

    void beginLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) override {}
    void analyze(edm::Event const& e, edm::EventSetup const& c) override;
    void endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) override;
  };

  // -----------------------------------------------------------------

  TestExpressLumiProducer::TestExpressLumiProducer(edm::ParameterSet const& ps) {
    consumes<LumiSummary, edm::InLumi>(edm::InputTag("expressLumiProducer", ""));
    consumes<LumiDetails, edm::InLumi>(edm::InputTag("expressLumiProducer", ""));
  }

  // -----------------------------------------------------------------

  void TestExpressLumiProducer::analyze(edm::Event const& e, edm::EventSetup const&) {}

  // -----------------------------------------------------------------

  void TestExpressLumiProducer::endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {
    Handle<LumiSummary> lumiSummary;
    lumiBlock.getByLabel("expressLumiProducer", lumiSummary);
    //std::cout<<"lumiSummary ptr "<<lumiSummary<<std::endl;
    if (lumiSummary->isValid()) {
      std::cout << *lumiSummary << "\n";
    } else {
      std::cout << "no valid lumi summary data" << std::endl;
    }
    Handle<LumiDetails> lumiDetails;
    lumiBlock.getByLabel("expressLumiProducer", lumiDetails);
    if (lumiDetails->isValid()) {
      std::cout << "valid detail" << std::endl;
      std::cout << "lumivalue bx 14 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 14) << std::endl;
      std::cout << "lumivalue bx 214 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 214) << std::endl;
      std::cout << "lumivalue bx 1475 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 1475) << std::endl;
      std::cout << "lumivalue bx 2775 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 2775) << std::endl;
      std::cout << "lumivalue bx 3500 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 3500) << std::endl;
    } else {
      std::cout << "no valid lumi detail data" << std::endl;
    }
  }
}  // namespace edmtest
using edmtest::TestExpressLumiProducer;

DEFINE_FWK_MODULE(TestExpressLumiProducer);
