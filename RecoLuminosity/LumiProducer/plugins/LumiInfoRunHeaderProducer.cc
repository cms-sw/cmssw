#include "RecoLuminosity/LumiProducer/plugins/LumiInfoRunHeaderProducer.h"
#include "DataFormats/Luminosity/interface/LumiInfoRunHeader.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"

LumiInfoRunHeaderProducer::LumiInfoRunHeaderProducer(const edm::ParameterSet& ps) {
  
  
  fillSchemeFromConfig_ = ps.getParameter<bool>("FillSchemeFromConfig");
  bunchSpacingFromConfig_ = ps.getParameter<int>("BunchSpacing");
  
  produces<LumiInfoRunHeader,edm::InRun>();
  
  
}

void LumiInfoRunHeaderProducer::endRunProduce(edm::Run &run, edm::EventSetup const& es) {
 
  int bunchspacing = 450;
  
  if (fillSchemeFromConfig_) {
    bunchspacing = bunchSpacingFromConfig_;
  }
  else {
    edm::RunNumber_t run = run.run();
    if (run == 178003 ||
        run == 178004 ||
        run == 209089 ||
        run == 209106 ||
        run == 209109 ||
        run == 209146 ||
        run == 209148 ||
        run == 209151) {
      bunchspacing = 25;
    }
    else {
      bunchspacing = 50;
    }    
  }
  
  std::auto_ptr<LumiInfoRunHeader> lumiInfoRunHeader(new LumiInfoRunHeader);
  lumiInfoRunHeader->setBunchSpacing(bunchspacing);
  run.put(fillSchemeInfo);  
  
}

DEFINE_FWK_MODULE(LumiInfoRunHeaderProducer);
