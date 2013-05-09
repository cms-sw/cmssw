#include "EventFilter/HcalRawToDigi/plugins/HcalLaserEventFiltProducer2012.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace std;

HcalLaserEventFiltProducer2012::HcalLaserEventFiltProducer2012(const edm::ParameterSet& iConfig) {
  hcalLaserEventFilter2012 = new HcalLaserEventFilter2012(iConfig);
  produces<bool>();
}

void HcalLaserEventFiltProducer2012::endJob() {
  hcalLaserEventFilter2012->endJob();
}

void HcalLaserEventFiltProducer2012::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::auto_ptr<bool> output(new bool(hcalLaserEventFilter2012->filter(iEvent, iSetup)));
  iEvent.put(output);
}

DEFINE_FWK_MODULE(HcalLaserEventFiltProducer2012);
