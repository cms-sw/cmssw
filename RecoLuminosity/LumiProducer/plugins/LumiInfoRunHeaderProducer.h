#ifndef RecoLuminosity_LumiProducer_LumiInfoRunHeaderProducer_h
#define RecoLuminosity_LumiProducer_LumiInfoRunHeaderProducer_h


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Luminosity/interface/LumiInfoRunHeader.h"
#include <vector>


class LumiInfoRunHeaderProducer : public edm::one::EDProducer<edm::BeginRunProducer> {
  public:
  
  LumiInfoRunHeaderProducer(const edm::ParameterSet&);  
  ~LumiInfoRunHeaderProducer() {}
    
  virtual void beginRunProduce(edm::Run &, edm::EventSetup const&) override;
  virtual void produce(edm::Event &, edm::EventSetup const&) override {}
    
  private:
    bool mcFillSchemeFromConfig_;
    bool mcFillSchemeFromDB_;
    int mcBunchSpacing_;
    
    edm::EDGetTokenT<LumiInfoRunHeader> lumiInfoRunHeaderMC_;
    
};
#endif


