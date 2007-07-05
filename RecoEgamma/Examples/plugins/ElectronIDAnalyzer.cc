#include "RecoEgamma/Examples/plugins/ElectronIDAnalyzer.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDAssociation.h"

ElectronIDAnalyzer::ElectronIDAnalyzer(const edm::ParameterSet& conf) : conf_(conf) {
  electronProducer_=conf.getParameter<std::string>("electronProducer");
  electronLabel_=conf.getParameter<std::string>("electronLabel");
  electronIDAssocProducer_ = conf.getParameter<std::string>("electronIDAssocProducer");
}

void ElectronIDAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& c) {

  // Read in electrons
  edm::Handle<reco::PixelMatchGsfElectronCollection> electrons;
  e.getByLabel(electronProducer_,electronLabel_,electrons);

  // Read in electron ID association map
  edm::Handle<reco::ElectronIDAssociationCollection> electronIDAssocHandle;
  e.getByLabel(electronIDAssocProducer_, electronIDAssocHandle);

  // Loop over electrons
  for (unsigned int i = 0; i < electrons->size(); i++){
    edm::Ref<reco::PixelMatchGsfElectronCollection> electronRef(electrons,i);
    // Find entry in electron ID map corresponding electron
    reco::ElectronIDAssociationCollection::const_iterator electronIDAssocItr;
    electronIDAssocItr = electronIDAssocHandle->find(electronRef);
    const reco::ElectronIDRef& id = electronIDAssocItr->val;
    bool cutBasedID = id->cutBasedDecision();
    std::cout << "Event " << e.id() << ", electron " << i+1 << ", cut based ID = " << cutBasedID << std::endl;
  }

}
