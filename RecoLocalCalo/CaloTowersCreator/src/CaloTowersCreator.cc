#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowersCreator.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

CaloTowersCreator::CaloTowersCreator(const edm::ParameterSet& conf) : 
  algo_(conf.getParameter<double>("EBThreshold"),
	conf.getParameter<double>("EEThreshold"),
	conf.getParameter<double>("HcalThreshold"),
	conf.getParameter<double>("HBThreshold"),
	conf.getParameter<double>("HESThreshold"),
	conf.getParameter<double>("HEDThreshold"),
	conf.getParameter<double>("HOThreshold"),
	conf.getParameter<double>("HF1Threshold"),
	conf.getParameter<double>("HF2Threshold"),
	conf.getParameter<double>("EBWeight"),
	conf.getParameter<double>("EEWeight"),
	conf.getParameter<double>("HBWeight"),
	conf.getParameter<double>("HESWeight"),
	conf.getParameter<double>("HEDWeight"),
	conf.getParameter<double>("HOWeight"),
	conf.getParameter<double>("HF1Weight"),
	conf.getParameter<double>("HF2Weight"),
	conf.getParameter<double>("EcutTower"),
	conf.getParameter<double>("EBSumThreshold"),
	conf.getParameter<double>("EESumThreshold"),
	conf.getParameter<bool>("UseHO")) 
{
  produces<CaloTowerCollection>();
}

void CaloTowersCreator::produce(edm::Event& e, const edm::EventSetup& c) {
  // get the necessary event setup objects...
  edm::ESHandle<CaloGeometry> pG;
  c.get<IdealGeometryRecord>().get(pG);
  
  algo_.setGeometry(&topo_,pG.product());
  
  // Step A: Get Inputs
  edm::Handle<HBHERecHitCollection> hbhe;
  edm::Handle<HORecHitCollection> ho;
  edm::Handle<HFRecHitCollection> hf;

  e.getByType(hbhe); // TODO : use selector
  e.getByType(ho);   // TODO : use selector
  e.getByType(hf);   // TODO : use selector

  // Step B: Create empty output
  std::auto_ptr<CaloTowerCollection> prod(new CaloTowerCollection());

  // Step C: Process
  algo_.create(*prod,*hbhe,*ho, *hf);

  // Step D: Put into the event
  e.put(prod);
}


#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CaloTowersCreator)
