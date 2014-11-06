#include "RecoLuminosity/LumiProducer/plugins/FillSchemeInfoProducer.h"
#include "DataFormats/Luminosity/interface/FillSchemeInfo.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"

FillSchemeInfoProducer::FillSchemeInfoProducer(const edm::ParameterSet& ps) {
  
  pileupSummaryInfos_ = consumes<std::vector<PileupSummaryInfo> >(edm::InputTag("addPileupInfo"));
  produces<FillSchemeInfo>();
  
  
}

void FillSchemeInfoProducer::produce(edm::Event &event, edm::EventSetup const& es) {
  
  int bunchspacing = 450;
  
  if (event.isRealData()) {
    edm::RunNumber_t run = event.run();
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
  else {
    edm::Handle<std::vector<PileupSummaryInfo> > pileupSummaryInfosH;
    event.getByToken(pileupSummaryInfos_,pileupSummaryInfosH);
    bunchspacing = pileupSummaryInfosH->front().getBunchSpacing();
  }
  
  
  std::auto_ptr<FillSchemeInfo> fillSchemeInfo(new FillSchemeInfo(bunchspacing));
  event.put(fillSchemeInfo);
  
  
}

DEFINE_FWK_MODULE(FillSchemeInfoProducer);
