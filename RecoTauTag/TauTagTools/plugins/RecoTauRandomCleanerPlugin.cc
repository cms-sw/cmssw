/*
 * RecoTauRandomCleanerPlugin
 *
 * Author: Evan K. Friis, UC Davis
 *
 * A reco tau cleaner plugin that selects a *random* PFTau.
 *
 */

#include <boost/functional/hash.hpp>
#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"


namespace reco { namespace tau {

class RecoTauRandomCleanerPlugin : public RecoTauCleanerPlugin {
  public:
    RecoTauRandomCleanerPlugin(const edm::ParameterSet& pset);
    virtual ~RecoTauRandomCleanerPlugin(){}
    // Get discriminant value for a given tau Ref
    double operator()(const reco::PFTauRef&) const;
  private:
    unsigned int seed_;
};

RecoTauRandomCleanerPlugin::RecoTauRandomCleanerPlugin(
    const edm::ParameterSet& pset):RecoTauCleanerPlugin(pset) {
  seed_ = pset.exists("seed") ? pset.getParameter<unsigned int>("seed") : 1234;
}

double RecoTauRandomCleanerPlugin::operator()(const reco::PFTauRef& tau) const {
  size_t output = seed_;
  boost::hash_combine(output, tau->pt());
  boost::hash_combine(output, tau->eta());
  return output;
}

}} // end namespace reco::tautools

// Register our plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory,
    reco::tau::RecoTauRandomCleanerPlugin,
    "RecoTauRandomCleanerPlugin");
