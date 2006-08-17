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
	true), // always uses HO!
  hbheLabel_(conf.getParameter<edm::InputTag>("hbheInput")),
  hoLabel_(conf.getParameter<edm::InputTag>("hoInput")),
  hfLabel_(conf.getParameter<edm::InputTag>("hfInput")),
  ecalLabels_(conf.getParameter<std::vector<edm::InputTag> >("ecalInputs")),
  allowMissingInputs_(conf.getUntrackedParameter<bool>("AllowMissingInputs",false))
{
  produces<CaloTowerCollection>();
}

void CaloTowersCreator::produce(edm::Event& e, const edm::EventSetup& c) {
  // get the necessary event setup objects...
  edm::ESHandle<CaloGeometry> pG;
  edm::ESHandle<HcalTopology> htopo;
  edm::ESHandle<CaloTowerConstituentsMap> cttopo;
  c.get<IdealGeometryRecord>().get(pG);
  c.get<IdealGeometryRecord>().get(htopo);
  c.get<IdealGeometryRecord>().get(cttopo);
  
  algo_.setGeometry(cttopo.product(),htopo.product(),pG.product());

  algo_.begin(); // clear the internal buffer
  
  // Step A/C: Get Inputs and process (repeatedly)
  try {
    edm::Handle<HBHERecHitCollection> hbhe;
    e.getByLabel(hbheLabel_,hbhe);
    algo_.process(*hbhe);
  } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) throw e;
  }

  try {
    edm::Handle<HORecHitCollection> ho;
    e.getByLabel(hoLabel_,ho);
    algo_.process(*ho);
  } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) throw e;
  }

  try {
    edm::Handle<HFRecHitCollection> hf;
    e.getByLabel(hfLabel_,hf);
    algo_.process(*hf);
  } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) throw e;
  }

  try {
    std::vector<edm::InputTag>::const_iterator i;
    for (i=ecalLabels_.begin(); i!=ecalLabels_.end(); i++) {
      edm::Handle<EcalRecHitCollection> ec;
      e.getByLabel(*i,ec);
      algo_.process(*ec);
    }
  } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) throw e;
  }

  // Step B: Create empty output
  std::auto_ptr<CaloTowerCollection> prod(new CaloTowerCollection());

  // Step C: Process
  algo_.finish(*prod);

  // Step D: Put into the event
  e.put(prod);
}


