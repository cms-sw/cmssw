#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/Common/interface/Association.h"
//#include "DataFormats/Common/interface/AssociativeIterator.h"

#include <boost/bind.hpp>
#include <boost/iterator/filter_iterator.hpp>

namespace tautools {

class RecoTauDistanceFromTruthPlugin : public reco::tau::RecoTauCleanerPlugin {
  public:
  RecoTauDistanceFromTruthPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector &&iC);
    ~RecoTauDistanceFromTruthPlugin() override {}
    double operator()(const reco::PFTauRef&) const override;
    void beginEvent() override;
  private:
    edm::InputTag matchingSrc_;
    typedef edm::Association<reco::GenJetCollection> GenJetAssociation;
    edm::Handle<GenJetAssociation> genTauMatch_;
};

RecoTauDistanceFromTruthPlugin::RecoTauDistanceFromTruthPlugin(
							       const edm::ParameterSet& pset, edm::ConsumesCollector &&iC): reco::tau::RecoTauCleanerPlugin(pset,std::move(iC)) {
  matchingSrc_ = pset.getParameter<edm::InputTag>("matching");
}

void RecoTauDistanceFromTruthPlugin::beginEvent() {
  // Load the matching information
  evt()->getByLabel(matchingSrc_, genTauMatch_);
}

double RecoTauDistanceFromTruthPlugin::operator()(const reco::PFTauRef& tauRef) const {

  GenJetAssociation::reference_type truth = (*genTauMatch_)[tauRef];

  // Check if the matching exists, if not return +infinity
  if (truth.isNull())
    return std::numeric_limits<double>::infinity();

  return std::abs(tauRef->pt() - truth->pt());
}

} // end tautools namespace


// Register our plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory, tautools::RecoTauDistanceFromTruthPlugin, "RecoTauDistanceFromTruthPlugin");
