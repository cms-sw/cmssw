#include "RecoTauTag/RecoTau/interface/RecoTauPluginsCommon.h"

namespace reco { namespace tau {

// ctor
RecoTauNamedPlugin::RecoTauNamedPlugin(const edm::ParameterSet& pset):
  name_(pset.getParameter<std::string>("name")) {}

const std::string& RecoTauNamedPlugin::name() const
{ return name_; }

// ctor
RecoTauEventHolderPlugin::RecoTauEventHolderPlugin(const edm::ParameterSet& pset)
  :RecoTauNamedPlugin(pset),evt_(NULL),es_(NULL) {}

const edm::Event* RecoTauEventHolderPlugin::evt() const { return evt_; }
edm::Event* RecoTauEventHolderPlugin::evt() { return evt_; }
const edm::EventSetup* RecoTauEventHolderPlugin::evtSetup() const { return es_; }

void RecoTauEventHolderPlugin::setup(edm::Event& evt, const edm::EventSetup& es)
{
  evt_ = &evt;
  es_ = &es;
  // Call the virtual beginEvent() function
  this->beginEvent();
}

} } // end reco::tau

