/*
 * RecoTauDiscriminantFromDiscriminator
 *
 * Makes a discriminator function from a PFRecoTauDiscriminator stored in the
 * event.
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */

#include <sstream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTauTag/RecoTau/interface/RecoTauDiscriminantPlugins.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

namespace reco::tau {

class RecoTauDiscriminantFromDiscriminator : public RecoTauDiscriminantPlugin{
  public:
    explicit RecoTauDiscriminantFromDiscriminator(
        const edm::ParameterSet& pset);
    void beginEvent() override;
    std::vector<double> operator()(const reco::PFTauRef& tau) const override;
  private:
    bool takeAbs_;
    double min_;
    double max_;
    typedef std::pair<edm::InputTag, edm::Handle<reco::PFTauDiscriminator> > DiscInfo;
    std::vector<DiscInfo> discriminators_;
};

RecoTauDiscriminantFromDiscriminator::RecoTauDiscriminantFromDiscriminator(
    const edm::ParameterSet& pset):RecoTauDiscriminantPlugin(pset) {

  takeAbs_ = pset.getParameter<bool>("takeAbs");
  min_ = pset.getParameter<double>("minValue");
  max_ = pset.getParameter<double>("maxValue");

  std::vector<edm::InputTag> discriminators =
    pset.getParameter<std::vector<edm::InputTag> >("discSrc");
  for(auto const& tag : discriminators) {
    discriminators_.push_back(std::make_pair(tag, edm::Handle<reco::PFTauDiscriminator>()));
  }
}

// Called by base class at the beginning of every event
void RecoTauDiscriminantFromDiscriminator::beginEvent() {
  for(auto& discInfo : discriminators_) {
    evt()->getByLabel(discInfo.first, discInfo.second);
  }
}

std::vector<double> RecoTauDiscriminantFromDiscriminator::operator()(
    const reco::PFTauRef& tau) const {
  edm::ProductID tauProdId = tau.id();
  double result = -999;
  bool foundGoodDiscriminator = false;
  for (size_t i = 0; i < discriminators_.size(); ++i) {
    // Check if the discriminator actually exists
    if (!discriminators_[i].second.isValid())
      continue;
    const reco::PFTauDiscriminator& disc = *(discriminators_[i].second);
    if (tauProdId == disc.keyProduct().id()) {
      foundGoodDiscriminator = true;
      result = (disc)[tau];
      break;
    }
  }
  // In case no discriminator is found.
  if (!foundGoodDiscriminator) {
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
  }
  if (result < min_)
    result = min_;
  if (result > max_)
    result = max_;
  return std::vector<double>(1, result);
}

} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauDiscriminantPluginFactory,
    reco::tau::RecoTauDiscriminantFromDiscriminator,
    "RecoTauDiscriminantFromDiscriminator");
