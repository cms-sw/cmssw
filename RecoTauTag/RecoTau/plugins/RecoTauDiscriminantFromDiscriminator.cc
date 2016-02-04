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
#include <sstream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
    // Check if the discriminator actually exists
    if (!discriminators_[i].second.isValid())
      continue;
    const reco::PFTauDiscriminator& disc = *(discriminators_[i].second);
    if (tauProdId == disc.keyProduct().id())
      return std::vector<double>(1, (disc)[tau]);
  }
  // Can only reach this point if not appropriate discriminator is defined for
  // the passed tau.
  std::stringstream error;
  error << "Couldn't find a PFTauDiscriminator usable with given tau."
    << std::endl << " Input tau has product id: " << tau.id() << std::endl;
  for (size_t i = 0; i < discriminators_.size(); ++i ) {
    error << "disc: " << discriminators_[i].first;
    error << " isValid: " << discriminators_[i].second.isValid();
    if (discriminators_[i].second.isValid()) {
      error << " product: " << discriminators_[i].second->keyProduct().id();
    }
    error << std::endl;
  }
  edm::LogError("BadDiscriminatorConfiguration") << error.str();
  return std::vector<double>(1,-999);
}

}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauDiscriminantPluginFactory,
    reco::tau::RecoTauDiscriminantFromDiscriminator,
    "RecoTauDiscriminantFromDiscriminator");
