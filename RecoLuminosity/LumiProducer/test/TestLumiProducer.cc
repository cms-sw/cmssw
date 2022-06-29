
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

  class TestLumiProducer : public edm::one::EDAnalyzer<edm::one::WatchLuminosityBlocks> {
  public:
    explicit TestLumiProducer(edm::ParameterSet const&);

    void beginLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) override {}
    void analyze(edm::Event const& e, edm::EventSetup const& c) override;
    void endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) override;
  };

  // -----------------------------------------------------------------

  TestLumiProducer::TestLumiProducer(edm::ParameterSet const& ps) {
    consumes<LumiSummary, edm::InLumi>(edm::InputTag("lumiProducer", ""));
    consumes<LumiDetails, edm::InLumi>(edm::InputTag("lumiProducer", ""));
  }

  // -----------------------------------------------------------------

  void TestLumiProducer::analyze(edm::Event const& e, edm::EventSetup const&) {}

  // -----------------------------------------------------------------

  void TestLumiProducer::endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {
    Handle<LumiSummary> lumiSummary;
    lumiBlock.getByLabel("lumiProducer", lumiSummary);
    if (lumiSummary->isValid()) {
      std::cout << *lumiSummary << "\n";
    } else {
      std::cout << "no valid lumi summary data" << std::endl;
    }
    Handle<LumiDetails> lumiDetails;
    lumiBlock.getByLabel("lumiProducer", lumiDetails);
    if (lumiDetails->isValid()) {
      //std::cout << *lumiDetails << "\n";
      std::cout << "lumivalue beamintensity 1 " << lumiDetails->lumiBeam1Intensity(1) << " "
                << lumiDetails->lumiBeam2Intensity(1) << std::endl;
      std::cout << "lumivalue 1 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 1) * 6.37 << " "
                << lumiDetails->lumiBeam1Intensity(1) << std::endl;
      std::cout << "lumivalue 214 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 214) * 6.37 << " "
                << lumiDetails->lumiBeam1Intensity(214) << std::endl;
      std::cout << "lumivalue 643 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 643) * 6.37 << " "
                << lumiDetails->lumiBeam1Intensity(643) << std::endl;
      std::cout << "lumivalue 895 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 895) * 6.37 << " "
                << lumiDetails->lumiBeam1Intensity(895) << std::endl;
      std::cout << "lumivalue 901 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 901) * 6.37 << " "
                << lumiDetails->lumiBeam1Intensity(901) << std::endl;
      std::cout << "lumivalue 1000 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 1000) * 6.37 << " "
                << lumiDetails->lumiBeam1Intensity(1000) << std::endl;
      std::cout << "lumivalue 1475 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 1475) * 6.37 << " "
                << lumiDetails->lumiBeam1Intensity(1475) << std::endl;
      std::cout << "lumivalue 2053 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 2053) * 6.37 << " "
                << lumiDetails->lumiBeam1Intensity(2053) << std::endl;
      std::cout << "lumivalue 2765 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 2765) * 6.37 << " "
                << lumiDetails->lumiBeam1Intensity(2765) << std::endl;
      std::cout << "lumivalue 3500 " << lumiDetails->lumiValue(LumiDetails::kOCC1, 3500) * 6.37 << " "
                << lumiDetails->lumiBeam1Intensity(3500) << std::endl;
    } else {
      std::cout << "no valid lumi detail data" << std::endl;
    }
    // We know the content we put into the objects in the
    // configuration, manually check to see that we can
    // retrieve the same values.

    // A small value to allow for machine precision variations when
    // comparing small numbers.
    /*
    double epsilon = 0.001;

    for (int i = 0; i < 5; ++i) {

      if ( (fabs(lumiDetails->lumiEtSum(i)     - (100.0 + i)) > epsilon) ||
           (fabs(lumiDetails->lumiEtSumErr(i)  - (200.0 + i)) > epsilon) ||
           (lumiDetails->lumiEtSumQual(i) != (300 + i)) ||
           (fabs(lumiDetails->lumiOcc(i)       - (400.0 + i)) > epsilon) ||
           (fabs(lumiDetails->lumiOccErr(i)    - (500.0 + i)) > epsilon) ||
           (lumiDetails->lumiOccQual(i) != (600 + i)) ) {
	std::cerr << "TestLumiProducer: Values read from LumiDetails object do not match input values (1)\n";
        abort();
      }    
      if ( (fabs(lumiDetails->lumiEtSum()[i]     - (100.0 + i)) > epsilon) ||
           (fabs(lumiDetails->lumiEtSumErr()[i]  - (200.0 + i)) > epsilon) ||
           (lumiDetails->lumiEtSumQual()[i] != (300 + i)) ||
           (fabs(lumiDetails->lumiOcc()[i]       - (400.0 + i)) > epsilon) ||
           (fabs(lumiDetails->lumiOccErr()[i]    - (500.0 + i)) > epsilon) ||
           (lumiDetails->lumiOccQual()[i] != (600 + i)) ) {
	std::cerr << "TestLumiProducer: Values read from LumiDetails object do not match input values (2)\n";
        abort();
      }    

      if ( (lumiSummary->l1RateCounter(i)  != (10 + i)) ||
           (lumiSummary->l1Scaler(i)       != (20 + i)) ||
           (lumiSummary->hltRateCounter(i) != (30 + i)) ||
           (lumiSummary->hltScaler(i)      != (40 + i)) ||
           (lumiSummary->hltInput(i)       != (50 + i)) ) {
        std::cerr << "TestLumiProducer: Values read from LumiSummary object do not match input values (1)\n";
        abort();
      }

      if ( (lumiSummary->l1RateCounter()[i]  != (10 + i)) ||
           (lumiSummary->l1Scaler()[i]       != (20 + i)) ||
           (lumiSummary->hltRateCounter()[i] != (30 + i)) ||
           (lumiSummary->hltScaler()[i]      != (40 + i)) ||
           (lumiSummary->hltInput()[i]       != (50 + i)) ) {
        std::cerr << "TestLumiProducer: Values read from LumiSummary object do not match input values (2)\n";
        abort();
      }
    }

    if ( (fabs(lumiSummary->avgInsDelLumi()    - 1.0)  > epsilon) ||
         (fabs(lumiSummary->avgInsDelLumiErr() - 2.0)  > epsilon) ||
         (fabs(lumiSummary->avgInsRecLumi()    - 1.0 * 0.95)  > epsilon) ||
         (fabs(lumiSummary->avgInsRecLumiErr() - 2.0 * 0.95)  > epsilon) ||
         (fabs(lumiSummary->deadFrac()      - 0.05) > epsilon) ||
         (fabs(lumiSummary->liveFrac()      - 0.95) > epsilon) ||
         (lumiSummary->lumiSecQual() != 3) ||
         (lumiSummary->lsNumber() != 5) ) {
      std::cerr << "TestLumiProducer: Values read from LumiSummary object do not match input values (3)\n";
      abort();
    }
    */
  }
}  // namespace edmtest

using edmtest::TestLumiProducer;

DEFINE_FWK_MODULE(TestLumiProducer);
