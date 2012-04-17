#include "FWCore/Framework/interface/EDAnalyzer.h"
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

namespace edmtest
{

  class TestExpressLumiProducer : public edm::EDAnalyzer
  {
  public:

    explicit TestExpressLumiProducer(edm::ParameterSet const&);
    virtual ~TestExpressLumiProducer();

    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
    virtual void endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c);
  };

  // -----------------------------------------------------------------

  TestExpressLumiProducer::TestExpressLumiProducer(edm::ParameterSet const& ps)
  {
  }

  // -----------------------------------------------------------------

  TestExpressLumiProducer::~TestExpressLumiProducer()
  {
  }

  // -----------------------------------------------------------------

  void TestExpressLumiProducer::analyze(edm::Event const& e,edm::EventSetup const&)
  {
  }

  // -----------------------------------------------------------------

  void TestExpressLumiProducer::endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {
    Handle<LumiSummary> lumiSummary;
    lumiBlock.getByLabel("expressLumiProducer", lumiSummary);
    if(lumiSummary->isValid()){
      std::cout << *lumiSummary << "\n";
    }else{
      std::cout << "no valid lumi summary data" <<std::endl;
    }
    Handle<LumiDetails> lumiDetails;
    lumiBlock.getByLabel("expressLumiProducer", lumiDetails);
    if(lumiDetails->isValid()){
      std::cout<<"valid detail"<<std::endl;
    }else{
      std::cout << "no valid lumi detail data" <<std::endl;
    }
  }
}//ns 
using edmtest::TestExpressLumiProducer;

DEFINE_FWK_MODULE(TestExpressLumiProducer);
