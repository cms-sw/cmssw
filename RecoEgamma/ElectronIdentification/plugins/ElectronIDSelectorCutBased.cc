#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorCutBased.h"

ElectronIDSelectorCutBased::ElectronIDSelectorCutBased (const edm::ParameterSet& conf) : conf_ (conf) 
{
  std::string algorithm_ = conf.getParameter<std::string> ("algorithm") ;
  
  if ( algorithm_ == "eIDCBClasses" )
     electronIDAlgo_ = new PTDRElectronID ();
  else if ( algorithm_ == "eIDCB" )
     electronIDAlgo_ = new CutBasedElectronID ();
  else 
  {
    edm::LogError("ElectronIDSelectorCutBased") << "Invalid algorithm parameter: must be eIDCBClasses or eIDCB." ;
    exit (1); 
  }
}

ElectronIDSelectorCutBased::~ElectronIDSelectorCutBased () 
{
  delete electronIDAlgo_ ;
}

void ElectronIDSelectorCutBased::newEvent (const edm::Event& e, const edm::EventSetup& c)
{
  electronIDAlgo_->setup (conf_);
}

double ElectronIDSelectorCutBased::operator () (const reco::GsfElectron & electron, const edm::Event& event) 
{
  return electronIDAlgo_->result (& (electron), event) ;
}
