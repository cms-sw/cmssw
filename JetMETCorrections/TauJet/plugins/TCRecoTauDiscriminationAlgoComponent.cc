/* 
 * class TCRecoTauDiscriminationAlgoComponent
 * created : May 4 2010,
 * revised : ,
 * Authors : Sami Lehti (HIP)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/TCTauAlgorithm.h"

using namespace reco;

class TCRecoTauDiscriminationAlgoComponent final : public CaloTauDiscriminationProducerBase {
    public:
      	explicit TCRecoTauDiscriminationAlgoComponent(const edm::ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig),    
	  tcTauAlgorithm(iConfig, consumesCollector()) {
      	}
      	~TCRecoTauDiscriminationAlgoComponent(){} 

      	double discriminate(const CaloTauRef& theCaloTauRef) const override;

	void beginEvent(const edm::Event&, const edm::EventSetup&) override;

    private:
	TCTauAlgorithm tcTauAlgorithm;
};

void TCRecoTauDiscriminationAlgoComponent::beginEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup){
	tcTauAlgorithm.eventSetup(iEvent,iSetup);
}


double TCRecoTauDiscriminationAlgoComponent::discriminate(const CaloTauRef& theCaloTauRef) const {
        auto algoused = TCTauAlgorithm::TCAlgoUndetermined;
	tcTauAlgorithm.recalculateEnergy(*theCaloTauRef, algoused);
	return algoused; // is this correct???  (elsewehre is  ? 1.:0.;)
}

DEFINE_FWK_MODULE(TCRecoTauDiscriminationAlgoComponent);
