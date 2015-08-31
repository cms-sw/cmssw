/*
 * RecoTauChargedHadronMultiplicityCleanerPlugin
 *
 * Author: Christian Veelken, NICPB Tallinn
 *
 * A reco tau cleaner plugin that ranks the PFTaus by the number of charged hadrons.
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

namespace reco { namespace tau {

class RecoTauChargedHadronMultiplicityCleanerPlugin : public RecoTauCleanerPlugin 
{
 public:
  RecoTauChargedHadronMultiplicityCleanerPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC);

  // Get ranking value for a given tau Ref
  double operator()(const reco::PFTauRef&) const override;
};

RecoTauChargedHadronMultiplicityCleanerPlugin::RecoTauChargedHadronMultiplicityCleanerPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
  : RecoTauCleanerPlugin(pset,std::move(iC)) 
{}

double RecoTauChargedHadronMultiplicityCleanerPlugin::operator()(const reco::PFTauRef& tau) const 
{
  // Get the ranking value for this tau. 
  // N.B. lower value means more "tau like"!
  double result = 0.;
  const std::vector<PFRecoTauChargedHadron>& chargedHadrons = tau->signalTauChargedHadronCandidates();
  for ( std::vector<PFRecoTauChargedHadron>::const_iterator chargedHadron = chargedHadrons.begin();
	chargedHadron != chargedHadrons.end(); ++chargedHadron ) {
    if      ( chargedHadron->algo() == PFRecoTauChargedHadron::kChargedPFCandidate ) result -= 8.;
    else if ( chargedHadron->algo() == PFRecoTauChargedHadron::kTrack              ) result -= 4.;
    else if ( chargedHadron->algo() == PFRecoTauChargedHadron::kPFNeutralHadron    ) result -= 2.;
    else                                                                             result -= 1.;
  }
  return result;
}

}} // end namespace reco::tau

// Register our plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory, 
    reco::tau::RecoTauChargedHadronMultiplicityCleanerPlugin, 
    "RecoTauChargedHadronMultiplicityCleanerPlugin");
