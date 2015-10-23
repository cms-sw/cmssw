/*
 * Original author: Alexander Nehrkorn (RWTH Aachen)
 *
 * Description:
 * This module rejects tau candidates that do not have unit charge.
 * It takes the fact into account that taus do not necessarily need
 * to be created from PF charged hadrons only but can be created
 * from a combination of PF charged hadrons and tracks.
 *
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

namespace reco { namespace tau {

class RecoTauChargeCleanerPlugin : public RecoTauCleanerPlugin
{
public:
	explicit RecoTauChargeCleanerPlugin(const edm::ParameterSet&, edm::ConsumesCollector &&iC);
	~RecoTauChargeCleanerPlugin() {}
	double operator()(const PFTauRef& tau) const override;

private:
	std::vector<unsigned> nprongs_;
	double failResult_;
	int charge_;
};

RecoTauChargeCleanerPlugin::RecoTauChargeCleanerPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector &&iC)
	: RecoTauCleanerPlugin(pset,std::move(iC)),
	  nprongs_(pset.getParameter<std::vector<unsigned> >("nprongs")),
	  failResult_(pset.getParameter<double>("selectionFailValue")),
	  charge_(pset.getParameter<int>("passForCharge"))
{}

double RecoTauChargeCleanerPlugin::operator()(const PFTauRef& cand) const
{
	int charge = 0;
	unsigned nChargedPFCandidate(0), nTrack(0);
	for(auto const& tauCand : cand->signalTauChargedHadronCandidates()){
		charge += tauCand.charge();
		if(tauCand.algoIs(reco::PFRecoTauChargedHadron::kChargedPFCandidate)) nChargedPFCandidate++;
		else if(tauCand.algoIs(reco::PFRecoTauChargedHadron::kTrack)) nTrack++;
	}

	for(auto nprong : nprongs_){
		if(nChargedPFCandidate+nTrack == nprong) return abs(charge)-charge_;
	}

	return failResult_;
}

}}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory, reco::tau::RecoTauChargeCleanerPlugin, "RecoTauChargeCleanerPlugin");
