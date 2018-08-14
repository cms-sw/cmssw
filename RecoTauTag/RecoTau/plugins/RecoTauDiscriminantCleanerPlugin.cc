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
  RecoTauDiscriminantCleanerPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector &&iC);
    ~RecoTauDiscriminantCleanerPlugin() override{}

    // Get discriminant value for a given tau Ref
    double operator()(const reco::PFTauRef&) const override;
    // Hook called from base class at the beginning of each event
    void beginEvent() override;

  private:
    edm::InputTag discriminatorSrc_;
    edm::Handle<PFTauDiscriminator> discriminator_;
  edm::EDGetTokenT<PFTauDiscriminator> discriminator_token;
};

RecoTauDiscriminantCleanerPlugin::RecoTauDiscriminantCleanerPlugin(
								   const edm::ParameterSet& pset, edm::ConsumesCollector &&iC):RecoTauCleanerPlugin(pset,std::move(iC)) {
  discriminatorSrc_ = pset.getParameter<edm::InputTag>("src");
  discriminator_token = iC.consumes<PFTauDiscriminator>(discriminatorSrc_);
}

void RecoTauDiscriminantCleanerPlugin::beginEvent() {
  // Load our handle to the discriminators from the event
  evt()->getByToken(discriminator_token, discriminator_);
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
