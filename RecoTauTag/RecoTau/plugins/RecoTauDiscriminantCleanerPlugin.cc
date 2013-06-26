/*
 * RecoTauDiscriminantCleanerPlugin
 *
 * Author: Evan K. Friis, UC Davis
 *
 * A reco tau cleaner plugin that given a PFTau returns the associated value 
 * stored in a PFTauDiscrimiantor.
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

namespace reco { namespace tau {

class RecoTauDiscriminantCleanerPlugin : public RecoTauCleanerPlugin {
  public:
    RecoTauDiscriminantCleanerPlugin(const edm::ParameterSet& pset);
    virtual ~RecoTauDiscriminantCleanerPlugin(){}

    // Get discriminant value for a given tau Ref
    double operator()(const reco::PFTauRef&) const;
    // Hook called from base class at the beginning of each event
    void beginEvent();

  private:
    edm::InputTag discriminatorSrc_;
    edm::Handle<PFTauDiscriminator> discriminator_;
};

RecoTauDiscriminantCleanerPlugin::RecoTauDiscriminantCleanerPlugin(
    const edm::ParameterSet& pset):RecoTauCleanerPlugin(pset) {
  discriminatorSrc_ = pset.getParameter<edm::InputTag>("src");
}

void RecoTauDiscriminantCleanerPlugin::beginEvent() {
  // Load our handle to the discriminators from the event
  evt()->getByLabel(discriminatorSrc_, discriminator_);
}

double RecoTauDiscriminantCleanerPlugin::operator()(
    const reco::PFTauRef& tau) const {
  // Get the discriminator result for this tau. N.B. result is negated!  lower 
  // = more "tau like"! This is opposite to the normal case.
  double result = -(*discriminator_)[tau];
  return result;
}

}} // end namespace reco::tau

// Register our plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory, 
    reco::tau::RecoTauDiscriminantCleanerPlugin, 
    "RecoTauDiscriminantCleanerPlugin");
