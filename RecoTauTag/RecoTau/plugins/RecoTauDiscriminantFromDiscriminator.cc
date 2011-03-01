/*
 * RecoTauDiscriminantFromDiscriminator
 *
 * Makes a discriminator function from a PFRecoTauDiscriminator stored in the
 * event.
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */

#include <boost/foreach.hpp>
#include "RecoTauTag/RecoTau/interface/RecoTauDiscriminantPlugins.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

namespace reco { namespace tau {

class RecoTauDiscriminantFromDiscriminator : public RecoTauDiscriminantPlugin{
  public:
    explicit RecoTauDiscriminantFromDiscriminator(
        const edm::ParameterSet& pset);
    void beginEvent();
    std::vector<double> operator()(const reco::PFTauRef& tau) const;
  private:
    typedef std::pair<edm::InputTag, edm::Handle<reco::PFTauDiscriminator> > DiscInfo;
    std::vector<DiscInfo> discriminators_;
};

RecoTauDiscriminantFromDiscriminator::RecoTauDiscriminantFromDiscriminator(
    const edm::ParameterSet& pset):RecoTauDiscriminantPlugin(pset) {
  if (pset.existsAs<edm::InputTag>("discSrc")) {
    discriminators_.push_back(std::make_pair(
          pset.getParameter<edm::InputTag>("discSrc"),
          edm::Handle<reco::PFTauDiscriminator>()));
  } else {
    // Get multiple discriminators.  This supports the case when the MVAHelper
    // class might be dealing with multiple tau collections (training)
    std::vector<edm::InputTag> discriminators =
      pset.getParameter<std::vector<edm::InputTag> >("discSrc");
    BOOST_FOREACH(const edm::InputTag& tag, discriminators) {
      discriminators_.push_back(std::make_pair(
            tag, edm::Handle<reco::PFTauDiscriminator>()));
    }
  }
}

// Called by base class at the beginning of every event
void RecoTauDiscriminantFromDiscriminator::beginEvent() {
  BOOST_FOREACH(DiscInfo& discInfo, discriminators_) {
    evt()->getByLabel(discInfo.first, discInfo.second);
  }
}

std::vector<double> RecoTauDiscriminantFromDiscriminator::operator()(
    const reco::PFTauRef& tau) const {
  edm::ProductID tauProdId = tau.id();
  for (size_t i = 0; i < discriminators_.size(); ++i) {
    const reco::PFTauDiscriminator& disc = *(discriminators_[i].second);
    if (tauProdId == disc.keyProduct().id())
      return std::vector<double>(1, (disc)[tau]);
  }
  // Can only reach this point if not appropriate discriminator is defined for
  // the passed tau.
  throw cms::Exception("NoDefinedDiscriminator")
    << "Couldn't find a PFTauDiscriminator usable with given tau."
    << " Input tau has product id: " << tau.id();

}

}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauDiscriminantPluginFactory,
    reco::tau::RecoTauDiscriminantFromDiscriminator,
    "RecoTauDiscriminantFromDiscriminator");
