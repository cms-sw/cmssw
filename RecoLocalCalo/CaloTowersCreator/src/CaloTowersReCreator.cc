#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowersReCreator.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

CaloTowersReCreator::CaloTowersReCreator(const edm::ParameterSet& conf) : 
  algo_(0.,0., false, false, false, false, 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,// thresholds cannot be reapplied
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
        conf.getParameter<int>("HcalPhase")
        ),
  allowMissingInputs_(false)
{
  tok_calo_ = consumes<CaloTowerCollection>(conf.getParameter<edm::InputTag>("caloLabel"));

  EBEScale=conf.getParameter<double>("EBEScale");
  EEEScale=conf.getParameter<double>("EEEScale");
  HBEScale=conf.getParameter<double>("HBEScale");
  HESEScale=conf.getParameter<double>("HESEScale");
  HEDEScale=conf.getParameter<double>("HEDEScale");
  HOEScale=conf.getParameter<double>("HOEScale");
  HF1EScale=conf.getParameter<double>("HF1EScale");
  HF2EScale=conf.getParameter<double>("HF2EScale");
  //  two notes:
  //  1) all this could go in a pset
  //  2) not clear the instanceLabel thing

  produces<CaloTowerCollection>();
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
  c.get<CaloGeometryRecord>().get(ctmap);

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
  e.getByToken(tok_calo_,calt);


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
    auto prod = std::make_unique<CaloTowerCollection>();

    // step C: rescale (without going threough metataowers)
    algo_.rescaleTowers(*calt, *prod);

    // Step D: Put into the event
    e.put(std::move(prod));
  }
}

void CaloTowersReCreator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("EBWeight", 1.0);
  desc.add<double>("HBEScale", 50.0);
  desc.add<double>("HEDWeight", 1.0);
  desc.add<double>("EEWeight", 1.0);
  desc.add<double>("HF1Weight", 1.0);
  desc.add<double>("HOWeight", 1.0);
  desc.add<double>("HESWeight", 1.0);
  desc.add<double>("HF2Weight", 1.0);
  desc.add<double>("HESEScale", 50.0);
  desc.add<double>("HEDEScale", 50.0);
  desc.add<double>("EBEScale", 50.0);
  desc.add<double>("HBWeight", 1.0);
  desc.add<double>("EEEScale", 50.0);
  desc.add<double>("MomHBDepth", 0.2);
  desc.add<double>("MomHEDepth", 0.4);
  desc.add<double>("MomEBDepth", 0.3);
  desc.add<double>("MomEEDepth", 0.0);
  desc.add<std::vector<double> >("HBGrid", {0.0, 2.0, 4.0, 5.0, 9.0, 20.0, 30.0, 50.0, 100.0, 1000.0});
  desc.add<std::vector<double> >("EEWeights", {0.51, 1.39, 1.71, 2.37, 2.32, 2.2, 2.1, 1.98, 1.8});
  desc.add<std::vector<double> >("HF2Weights", {1.0, 1.0, 1.0, 1.0, 1.0});
  desc.add<std::vector<double> >("HOWeights", {1.0, 1.0, 1.0, 1.0, 1.0});
  desc.add<std::vector<double> >("EEGrid", {2.0, 4.0, 5.0, 9.0, 20.0, 30.0, 50.0, 100.0, 300.0});
  desc.add<std::vector<double> >("HBWeights", {2.0, 1.86, 1.69, 1.55, 1.37, 1.19, 1.13, 1.11, 1.09, 1.0});
  desc.add<std::vector<double> >("HF2Grid", {-1.0, 1.0, 10.0, 100.0, 1000.0});
  desc.add<std::vector<double> >("HEDWeights", {1.7, 1.57, 1.54, 1.49, 1.41, 1.26, 1.19, 1.15, 1.12, 1.0});
  desc.add<std::vector<double> >("HF1Grid", {-1.0, 1.0, 10.0, 100.0, 1000.0});
  desc.add<std::vector<double> >("EBWeights", {0.86, 1.47, 1.66, 2.01, 1.98, 1.86, 1.83, 1.74, 1.65});
  desc.add<std::vector<double> >("HF1Weights", {1.0, 1.0, 1.0, 1.0, 1.0});
  desc.add<std::vector<double> >("HESGrid", {0.0, 2.0, 4.0, 5.0, 9.0, 20.0, 30.0, 50.0, 100.0, 1000.0});
  desc.add<std::vector<double> >("HESWeights", {1.7, 1.57, 1.54, 1.49, 1.41, 1.26, 1.19, 1.15, 1.12, 1.0});
  desc.add<std::vector<double> >("HEDGrid", {0.0, 2.0, 4.0, 5.0, 9.0, 20.0, 30.0, 50.0, 100.0, 1000.0});
  desc.add<std::vector<double> >("HOGrid", {-1.0, 1.0, 10.0, 100.0, 1000.0});
  desc.add<std::vector<double> >("EBGrid", {2.0, 4.0, 5.0, 9.0, 20.0, 30.0, 50.0, 100.0, 300.0});
  desc.add<edm::InputTag>("caloLabel", edm::InputTag("calotowermaker"));
  desc.add<int>("MomConstrMethod", 1);
  desc.add<int>("HcalPhase", 0);

  descriptions.addDefault(desc);
}
