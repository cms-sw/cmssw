/* 
 * class TCRecoTauDiscriminationAgainstHadronicJets
 * created : April 21 2010,
 * revised : ,
 * Authors : Sami Lehti (HIP)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/TCTauAlgorithm.h"

using namespace reco;

class TCRecoTauDiscriminationAgainstHadronicJets final  : public CaloTauDiscriminationProducerBase {
    public:
      	explicit TCRecoTauDiscriminationAgainstHadronicJets(const edm::ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig),
	  tcTauAlgorithm(iConfig, consumesCollector()) {
      	}
      	~TCRecoTauDiscriminationAgainstHadronicJets(){} 

      	double discriminate(const CaloTauRef& theCaloTauRef) const override;
	void beginEvent(const edm::Event&, const edm::EventSetup&) override;

    private:
	TCTauAlgorithm  tcTauAlgorithm;
};

void TCRecoTauDiscriminationAgainstHadronicJets::beginEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup){
	tcTauAlgorithm.eventSetup(iEvent,iSetup);
}


double TCRecoTauDiscriminationAgainstHadronicJets::discriminate(const CaloTauRef& theCaloTauRef) const {
        auto algoused = TCTauAlgorithm::TCAlgoUndetermined;
	tcTauAlgorithm.recalculateEnergy(*theCaloTauRef, algoused);
	return (algoused != TCTauAlgorithm::TCAlgoHadronicJet) ? 1. : 0.;
}

DEFINE_FWK_MODULE(TCRecoTauDiscriminationAgainstHadronicJets);
