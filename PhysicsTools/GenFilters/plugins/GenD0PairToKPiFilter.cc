// File: GenD0PairToKPiFilter.cc

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include <vector>

class GenD0PairToKPiFilter : public edm::stream::EDFilter<> {
	public:
		explicit GenD0PairToKPiFilter(const edm::ParameterSet&);
		bool filter(edm::Event&, const edm::EventSetup&) override;

	private:
		edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken_;
};

GenD0PairToKPiFilter::GenD0PairToKPiFilter(const edm::ParameterSet& iConfig) {
	genParticlesToken_ = consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("src"));
}

bool GenD0PairToKPiFilter::filter(edm::Event& iEvent, const edm::EventSetup&) {
	edm::Handle<reco::GenParticleCollection> genParticles;
	iEvent.getByToken(genParticlesToken_, genParticles);

	int D_count=0;


	for (const auto& p : *genParticles) {
		if (std::abs(p.pdgId()) != 421 || p.numberOfDaughters() != 2)
			continue;
		if (abs(p.pt()) < 1) continue; 

		const reco::Candidate* d1 = p.daughter(0);
		const reco::Candidate* d2 = p.daughter(1);

		int pdg1 = d1->pdgId();
		int pdg2 = d2->pdgId();

		//std::cout << "-----------------------------------------------" << std::endl;

		if (p.pdgId() == 421) {  // D0
			if (((pdg1 == -321 && pdg2 == 211) || (pdg1 == 211 && pdg2 == -321)) && abs(d1->eta()) < 2.6 && abs(d2->eta()) < 2.6)
			{       
				D_count++;
			}
		}
		
		if (p.pdgId() == -421) {  // D0bar
			if (((pdg1 == 321 && pdg2 == -211) || (pdg1 == -211 && pdg2 == 321)) && abs(d1->eta()) < 2.6 && abs(d2->eta()) < 2.6)
			{      
				D_count++;
			}
		}
		if (D_count >= 2) return true;

	}

	return false;
}

DEFINE_FWK_MODULE(GenD0PairToKPiFilter);
