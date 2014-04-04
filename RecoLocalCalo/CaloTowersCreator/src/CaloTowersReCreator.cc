#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowersReCreator.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoLocalCalo/CaloTowersCreator/interface/ctEScales.h"

CaloTowersReCreator::CaloTowersReCreator(const edm::ParameterSet& conf) : 
  algo_(0.,0., false, false, false, false, 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0., // thresholds cannot be reapplied
        conf.getParameter<std::vector<double> >("EBGrid"),
        conf.getParameter<std::vector<double> >("EBWeights"),
        conf.getParameter<std::vector<double> >("EEGrid"),
        conf.getParameter<std::vector<double> >("EEWeights"),
        conf.getParameter<std::vector<double> >("HBGrid"),
        conf.getParameter<std::vector<double> >("HBWeights"),
        conf.getParameter<std::vector<double> >("HESGrid"),
        conf.getParameter<std::vector<double> >("HESWeights"),
        conf.getParameter<std::vector<double> >("HEDGrid"),
        conf.getParameter<std::vector<double> >("HEDWeights"),
        conf.getParameter<std::vector<double> >("HOGrid"),
        conf.getParameter<std::vector<double> >("HOWeights"),
        conf.getParameter<std::vector<double> >("HF1Grid"),
        conf.getParameter<std::vector<double> >("HF1Weights"),
        conf.getParameter<std::vector<double> >("HF2Grid"),
        conf.getParameter<std::vector<double> >("HF2Weights"),
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
        conf.getParameter<double>("MomHBDepth"),
        conf.getParameter<double>("MomHEDepth"),
        conf.getParameter<double>("MomEBDepth"),
        conf.getParameter<double>("MomEEDepth"),
        conf.getParameter<double>("HcalPhase")
        ),
  caloLabel_(conf.getParameter<edm::InputTag>("caloLabel")),
  allowMissingInputs_(false)
{
  EBEScale=conf.getParameter<double>("EBEScale");
  EEEScale=conf.getParameter<double>("EEEScale");
  HBEScale=conf.getParameter<double>("HBEScale");
  HESEScale=conf.getParameter<double>("HESEScale");
  HEDEScale=conf.getParameter<double>("HEDEScale");
  HOEScale=conf.getParameter<double>("HOEScale");
  HF1EScale=conf.getParameter<double>("HF1EScale");
  HF2EScale=conf.getParameter<double>("HF2EScale");
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
  edm::ESHandle<CaloTowerTopology> cttopo;
  edm::ESHandle<CaloTowerConstituentsMap> ctmap;
  c.get<CaloGeometryRecord>().get(pG);
  c.get<HcalRecNumberingRecord>().get(htopo);
  c.get<HcalRecNumberingRecord>().get(cttopo);
  c.get<HcalRecNumberingRecord>().get(ctmap);
 
  algo_.setEBEScale(EBEScale);
  algo_.setEEEScale(EEEScale);
  algo_.setHBEScale(HBEScale);
  algo_.setHESEScale(HESEScale);
  algo_.setHEDEScale(HEDEScale);
  algo_.setHOEScale(HOEScale);
  algo_.setHF1EScale(HF1EScale);
  algo_.setHF2EScale(HF2EScale);
  algo_.setGeometry(cttopo.product(),ctmap.product(),htopo.product(),pG.product());

  algo_.begin(); // clear the internal buffer
  
  // Step A/C: Get Inputs and process (repeatedly)
  edm::Handle<CaloTowerCollection> calt;
  e.getByLabel(caloLabel_,calt);

/*
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
*/

  // modified to rescale the CaloTowers directly
  // without going through metatowers
  // required for the algorithms that make use of individual
  // crystal information

  if (!calt.isValid()) {
    // can't find it!
    if (!allowMissingInputs_) {
      *calt;  // will throw the proper exception
    }
  } else {
    // Step B: Create empty output
    std::auto_ptr<CaloTowerCollection> prod(new CaloTowerCollection());

    // step C: rescale (without going threough metataowers)
    algo_.rescaleTowers(*calt, *prod);

    // Step D: Put into the event
    if (ctEScales.instanceLabel=="") e.put(prod);
    else e.put(prod,ctEScales.instanceLabel);
  }

}

