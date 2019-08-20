#include <vector>

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/Selector.hh"

using namespace std;
using namespace fastjet;

class LHEJetFilter : public edm::EDFilter
{
	public:
		explicit LHEJetFilter(const edm::ParameterSet&);
		virtual ~LHEJetFilter() {}

	private:
		bool filter(edm::Event&, const edm::EventSetup&) override;

		edm::EDGetTokenT<LHEEventProduct> tokenLHEEvent_; 
		double jetPtMin_;
		JetDefinition jetdef_;

		vector<PseudoJet> jetconsts_;


};
 
LHEJetFilter::LHEJetFilter(const edm::ParameterSet& params)
	: tokenLHEEvent_(consumes<LHEEventProduct>(params.getParameter<edm::InputTag>("src"))),
	jetPtMin_(params.getParameter<double>("jetPtMin")),
	jetdef_(antikt_algorithm, params.getParameter<double>("jetR"))
{

}

bool LHEJetFilter::filter(edm::Event& evt, const edm::EventSetup& params)
{
	edm::Handle<LHEEventProduct> lheinfo;
	evt.getByToken(tokenLHEEvent_, lheinfo);

	if(!lheinfo.isValid()) {return true;}

	jetconsts_.clear();
	const lhef::HEPEUP& hepeup = lheinfo->hepeup();
	for(size_t p = 0 ; p < hepeup.IDUP.size() ; ++p)
	{
		if(hepeup.ISTUP[p] == 1)
		{
			const lhef::HEPEUP::FiveVector& mom = hepeup.PUP[p];
			jetconsts_.emplace_back(mom[0], mom[1], mom[2], mom[3]);
		}

	}

	ClusterSequence cs(jetconsts_, jetdef_);
	vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets());

	if(jets.size() == 0) {return false;}

	return jets[0].perp() > jetPtMin_;
}
 
 // Define module as a plug-in
 DEFINE_FWK_MODULE(LHEJetFilter);
