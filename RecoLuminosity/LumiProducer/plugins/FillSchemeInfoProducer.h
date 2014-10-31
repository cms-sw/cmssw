#ifndef RecoLuminosity_LumiProducer_FillSchemeInfoProducer_h
#define RecoLuminosity_LumiProducer_FillSchemeInfoProducer_h


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include <vector>


class FillSchemeInfoProducer : public edm::one::EDProducer<> {
  public:
  
  FillSchemeInfoProducer(const edm::ParameterSet&);  
  ~FillSchemeInfoProducer() {}
    
  virtual void produce(edm::Event &, edm::EventSetup const&) override;
    
  private:
    edm::EDGetTokenT<std::vector<PileupSummaryInfo> > pileupSummaryInfos_; 

};
#endif


