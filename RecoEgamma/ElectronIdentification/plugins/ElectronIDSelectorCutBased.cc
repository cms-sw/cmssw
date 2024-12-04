#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorCutBased.h"

#include "ClassBasedElectronID.h"
#include "CutBasedElectronID.h"
#include "PTDRElectronID.h"

ElectronIDSelectorCutBased::ElectronIDSelectorCutBased(const edm::ParameterSet& conf, edm::ConsumesCollector& iC) {
  std::string algorithm = conf.getParameter<std::string>("algorithm");

  if (algorithm == "eIDClassBased")
    electronIDAlgo_ = std::make_unique<ClassBasedElectronID>(conf);
  else if (algorithm == "eIDCBClasses")
    electronIDAlgo_ = std::make_unique<PTDRElectronID>(conf);
  else if (algorithm == "eIDCB")
    electronIDAlgo_ = std::make_unique<CutBasedElectronID>(conf, iC);
  else {
    throw cms::Exception("Configuration")
        << "Invalid algorithm parameter in ElectronIDSelectorCutBased: must be eIDCBClasses or eIDCB.";
  }
}

double ElectronIDSelectorCutBased::operator()(const reco::GsfElectron& ele,
                                              const edm::Event& e,
                                              const edm::EventSetup& es) const {
  return electronIDAlgo_->result(&(ele), e, es);
}
