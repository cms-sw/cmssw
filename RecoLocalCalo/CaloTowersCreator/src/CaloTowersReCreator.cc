#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowersReCreator.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoLocalCalo/CaloTowersCreator/interface/ctEScales.h"

CaloTowersReCreator::CaloTowersReCreator(const edm::ParameterSet& conf) : 
  algo_(0.,0.,0.,0.,0.,0.,0.,0.,0., // thresholds cannot be reapplied
        conf.getUntrackedParameter<std::vector<double> >("EBGrid",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("EBWeights",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("EEGrid",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("EEWeights",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("HBGrid",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("HBWeights",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("HESGrid",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("HESWeights",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("HEDGrid",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("HEDWeights",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("HOGrid",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("HOWeights",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("HF1Grid",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("HF1Weights",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("HF2Grid",std::vector<double>(10,0.)),
        conf.getUntrackedParameter<std::vector<double> >("HF2Weights",std::vector<double>(10,0.)),
        conf.getParameter<double>("EBWeight"),
        conf.getParameter<double>("EEWeight"),
        conf.getParameter<double>("HBWeight"),
        conf.getParameter<double>("HESWeight"),
        conf.getParameter<double>("HEDWeight"),
        conf.getParameter<double>("HOWeight"),
        conf.getParameter<double>("HF1Weight"),
        conf.getParameter<double>("HF2Weight"),
        0.,0.,0.,
        conf.getParameter<bool>("UseHO"),
        // (these have no effect on recreation: here for compatibility)
        conf.getParameter<int>("MomConstrMethod"),
        conf.getParameter<double>("MomEmDepth"),
        conf.getParameter<double>("MomHadDepth"),
        conf.getParameter<double>("MomTotDepth")
        ),
  caloLabel_(conf.getParameter<edm::InputTag>("caloLabel")),
  allowMissingInputs_(false)
{
  EBEScale=conf.getUntrackedParameter<double>("EBEScale",50.);
  EEEScale=conf.getUntrackedParameter<double>("EEEScale",50.);
  HBEScale=conf.getUntrackedParameter<double>("HBEScale",50.);
  HESEScale=conf.getUntrackedParameter<double>("HESEScale",50.);
  HEDEScale=conf.getUntrackedParameter<double>("HEDEScale",50.);
  HOEScale=conf.getUntrackedParameter<double>("HOEScale",50.);
  HF1EScale=conf.getUntrackedParameter<double>("HF1EScale",50.);
  HF2EScale=conf.getUntrackedParameter<double>("HF2EScale",50.);
  if (ctEScales.instanceLabel=="") produces<CaloTowerCollection>();
  else produces<CaloTowerCollection>(ctEScales.instanceLabel);
  //  two notes:
  //  1) all this could go in a pset
  //  2) not clear the instanceLabel thing
}

void CaloTowersReCreator::produce(edm::Event& e, const edm::EventSetup& c) {
  // get the necessary event setup objects...
  edm::ESHandle<CaloGeometry> pG;
  edm::ESHandle<HcalTopology> htopo;
  edm::ESHandle<CaloTowerConstituentsMap> cttopo;
  c.get<CaloGeometryRecord>().get(pG);
  c.get<IdealGeometryRecord>().get(htopo);
  c.get<IdealGeometryRecord>().get(cttopo);
 
  algo_.setEBEScale(EBEScale);
  algo_.setEEEScale(EEEScale);
  algo_.setHBEScale(HBEScale);
  algo_.setHESEScale(HESEScale);
  algo_.setHEDEScale(HEDEScale);
  algo_.setHOEScale(HOEScale);
  algo_.setHF1EScale(HF1EScale);
  algo_.setHF2EScale(HF2EScale);
  algo_.setGeometry(cttopo.product(),htopo.product(),pG.product());

  algo_.begin(); // clear the internal buffer
  
  // Step A/C: Get Inputs and process (repeatedly)
  edm::Handle<CaloTowerCollection> calt;
  e.getByLabel(caloLabel_,calt);

  if (!calt.isValid()) {
    // can't find it!
    if (!allowMissingInputs_) {
      *calt;  // will throw the proper exception
    }
  } else {
    algo_.process(*calt);
  }

  // Step B: Create empty output
  std::auto_ptr<CaloTowerCollection> prod(new CaloTowerCollection());

  // Step C: Process
  algo_.finish(*prod);

  // Step D: Put into the event
  if (ctEScales.instanceLabel=="") e.put(prod);
  else e.put(prod,ctEScales.instanceLabel);
}

