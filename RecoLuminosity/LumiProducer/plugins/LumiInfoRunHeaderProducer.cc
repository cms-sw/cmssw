#include "RecoLuminosity/LumiProducer/plugins/LumiInfoRunHeaderProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"

LumiInfoRunHeaderProducer::LumiInfoRunHeaderProducer(const edm::ParameterSet& ps) {
  
  
  mcFillSchemeFromConfig_ = ps.getParameter<bool>("MCFillSchemeFromConfig");
  mcFillSchemeFromDB_ = ps.getParameter<bool>("MCFillSchemeFromDB");
  mcBunchSpacing_ = ps.getParameter<int>("MCBunchSpacing");
  
  lumiInfoRunHeaderMC_ = consumes<LumiInfoRunHeader,edm::InRun>(edm::InputTag("lumiInfoRunHeaderMC"));
  produces<LumiInfoRunHeader,edm::InRun>();
  
  
}

void LumiInfoRunHeaderProducer::beginRunProduce(edm::Run &run, edm::EventSetup const& es) {
   
  //this is a hack, replace with proper call
  bool isRealData = (run.runAuxiliary().beginTime().unixTime()>0);
  
  if (isRealData || mcFillSchemeFromDB_) {
    int bunchspacing = 450;
    
    //this is a placeholder hardcoded list which should be replaced with proper access from lumidb/wherever
    if (run.run() == 178003 ||
        run.run() == 178004 ||
        run.run() == 209089 ||
        run.run() == 209106 ||
        run.run() == 209109 ||
        run.run() == 209146 ||
        run.run() == 209148 ||
        run.run() == 209151) {
      bunchspacing = 25;
    }
    else {
      bunchspacing = 50;
    }    
    
    std::auto_ptr<LumiInfoRunHeader> lumiInfoRunHeader(new LumiInfoRunHeader);
    lumiInfoRunHeader->setBunchSpacing(bunchspacing);
    run.put(lumiInfoRunHeader);  
  }
  else if (mcFillSchemeFromConfig_) {
    std::auto_ptr<LumiInfoRunHeader> lumiInfoRunHeader(new LumiInfoRunHeader);
    lumiInfoRunHeader->setBunchSpacing(mcBunchSpacing_);
    run.put(lumiInfoRunHeader);      
  }
  else {
    //When running on MC when not configured as lumiInfoRunHeaderMC source, then simply copy existing lumiInfoRunHeaderMC product
    edm::Handle<LumiInfoRunHeader> lumiInfoRunHeaderMCH;
    run.getByToken(lumiInfoRunHeaderMC_,lumiInfoRunHeaderMCH);
    
    std::auto_ptr<LumiInfoRunHeader> lumiInfoRunHeader(new LumiInfoRunHeader(*lumiInfoRunHeaderMCH));
    run.put(lumiInfoRunHeader);     
  }
  
  
}

DEFINE_FWK_MODULE(LumiInfoRunHeaderProducer);
