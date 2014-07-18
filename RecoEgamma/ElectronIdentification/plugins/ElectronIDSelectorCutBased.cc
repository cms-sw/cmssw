#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorCutBased.h"

ElectronIDSelectorCutBased::ElectronIDSelectorCutBased (const edm::ParameterSet& conf, edm::ConsumesCollector & iC) : conf_ (conf)
{
  std::string algorithm_ = conf.getParameter<std::string> ("algorithm") ;

  if ( algorithm_ == "eIDClassBased" )
     electronIDAlgo_ = new ClassBasedElectronID ();
  else if ( algorithm_ == "eIDCBClasses" )
     electronIDAlgo_ = new PTDRElectronID ();
  else if ( algorithm_ == "eIDCB" )
    electronIDAlgo_ = new CutBasedElectronID (conf,iC);
  else {
    throw cms::Exception("Configuration")
      << "Invalid algorithm parameter in ElectronIDSelectorCutBased: must be eIDCBClasses or eIDCB." ;
  }
}

ElectronIDSelectorCutBased::~ElectronIDSelectorCutBased ()
{
  delete electronIDAlgo_ ;
}

void ElectronIDSelectorCutBased::newEvent (const edm::Event& e, const edm::EventSetup& es)
{
  electronIDAlgo_->setup (conf_);
}

double ElectronIDSelectorCutBased::operator () (const reco::GsfElectron & ele, const edm::Event& e, const edm::EventSetup& es)
{
  return electronIDAlgo_->result (& (ele), e, es) ;
}
