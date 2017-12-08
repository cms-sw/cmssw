/*
 * RecoGenericTauDiscriminantCleanerPlugin
 *
 * Author: Evan K. Friis, UC Davis
 *
 * A reco tau cleaner plugin that given a PFTau returns the associated value 
 * stored in a PFTauDiscrimiantor.
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/PFBaseTauDiscriminator.h"

namespace reco { namespace tau {

template<class TauType, class DiscriminatorType>
class RecoGenericTauDiscriminantCleanerPlugin : public RecoTauCleanerPlugin<TauType> {
  public:
    RecoGenericTauDiscriminantCleanerPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector &&iC);
    ~RecoGenericTauDiscriminantCleanerPlugin() override{}

    // Get discriminant value for a given tau Ref
    double operator()(const edm::Ref<std::vector<TauType> >&) const override;
    // Hook called from base class at the beginning of each event
    void beginEvent() override;

  private:
    edm::InputTag discriminatorSrc_;
    edm::Handle<DiscriminatorType> discriminator_;
  edm::EDGetTokenT<DiscriminatorType> discriminator_token;
};

template<class TauType, class DiscriminatorType>
RecoGenericTauDiscriminantCleanerPlugin<TauType, DiscriminatorType>::RecoGenericTauDiscriminantCleanerPlugin(
								   const edm::ParameterSet& pset, edm::ConsumesCollector &&iC):RecoTauCleanerPlugin<TauType>(pset,std::move(iC)) {
  discriminatorSrc_ = pset.getParameter<edm::InputTag>("src");
  discriminator_token = iC.consumes<DiscriminatorType>(discriminatorSrc_);
}

template<class TauType, class DiscriminatorType>
void RecoGenericTauDiscriminantCleanerPlugin<TauType, DiscriminatorType>::beginEvent() {
  // Load our handle to the discriminators from the event
  this->evt()->getByToken(discriminator_token, discriminator_);
}

template<class TauType, class DiscriminatorType>
double RecoGenericTauDiscriminantCleanerPlugin<TauType, DiscriminatorType>::operator()(
    const edm::Ref<std::vector<TauType> >& tau) const {
  // Get the discriminator result for this tau. N.B. result is negated!  lower 
  // = more "tau like"! This is opposite to the normal case.
  double result = -(*discriminator_)[tau];
  return result;
}

template class RecoGenericTauDiscriminantCleanerPlugin<reco::PFTau, reco::PFTauDiscriminator>;
typedef RecoGenericTauDiscriminantCleanerPlugin<reco::PFTau, reco::PFTauDiscriminator> RecoTauDiscriminantCleanerPlugin;

template class RecoGenericTauDiscriminantCleanerPlugin<reco::PFBaseTau, reco::PFBaseTauDiscriminator>;
typedef RecoGenericTauDiscriminantCleanerPlugin<reco::PFBaseTau, reco::PFBaseTauDiscriminator> RecoBaseTauDiscriminantCleanerPlugin;

}} // end namespace reco::tau

// Register our plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory, 
    reco::tau::RecoTauDiscriminantCleanerPlugin, 
    "RecoTauDiscriminantCleanerPlugin");
DEFINE_EDM_PLUGIN(RecoBaseTauCleanerPluginFactory, 
    reco::tau::RecoBaseTauDiscriminantCleanerPlugin, 
    "RecoBaseTauDiscriminantCleanerPlugin");

