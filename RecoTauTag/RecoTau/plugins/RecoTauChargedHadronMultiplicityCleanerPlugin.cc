/*
 * RecoTauChargedHadronMultiplicityCleanerPlugin
 *
 * Author: Christian Veelken, NICPB Tallinn
 *
 * A reco tau cleaner plugin that ranks the PFTaus by the number of charged hadrons.
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"

namespace reco { namespace tau {

template<class TauType>
class RecoGenericTauChargedHadronMultiplicityCleanerPlugin : public RecoTauCleanerPlugin<TauType> 
{
 public:
  RecoGenericTauChargedHadronMultiplicityCleanerPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC);

  // Get ranking value for a given tau Ref
  double operator()(const edm::Ref<std::vector<TauType> >&) const override;
};

template<class TauType>
RecoGenericTauChargedHadronMultiplicityCleanerPlugin<TauType>::RecoGenericTauChargedHadronMultiplicityCleanerPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
  : RecoTauCleanerPlugin<TauType>(pset,std::move(iC)) 
{}

template<class TauType>
double RecoGenericTauChargedHadronMultiplicityCleanerPlugin<TauType>::operator()(const edm::Ref<std::vector<TauType> >& tau) const 
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

template class RecoGenericTauChargedHadronMultiplicityCleanerPlugin<reco::PFTau>;
typedef RecoGenericTauChargedHadronMultiplicityCleanerPlugin<reco::PFTau> RecoTauChargedHadronMultiplicityCleanerPlugin;

template class RecoGenericTauChargedHadronMultiplicityCleanerPlugin<reco::PFBaseTau>;
typedef RecoGenericTauChargedHadronMultiplicityCleanerPlugin<reco::PFBaseTau> RecoBaseTauChargedHadronMultiplicityCleanerPlugin;

}} // end namespace reco::tau

// Register our plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory, 
    reco::tau::RecoTauChargedHadronMultiplicityCleanerPlugin, 
    "RecoTauChargedHadronMultiplicityCleanerPlugin");
DEFINE_EDM_PLUGIN(RecoBaseTauCleanerPluginFactory, 
    reco::tau::RecoBaseTauChargedHadronMultiplicityCleanerPlugin, 
    "RecoBaseTauChargedHadronMultiplicityCleanerPlugin");
