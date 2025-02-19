/* 
 * class TCRecoTauDiscriminationAlgoComponent
 * created : May 4 2010,
 * revised : ,
 * Authors : Sami Lehti (HIP)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/TCTauAlgorithm.h"

using namespace reco;

class TCRecoTauDiscriminationAlgoComponent : public CaloTauDiscriminationProducerBase {
    public:
      	explicit TCRecoTauDiscriminationAlgoComponent(const edm::ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig){   
		tcTauAlgorithm = new TCTauAlgorithm(iConfig);
      	}
      	~TCRecoTauDiscriminationAlgoComponent(){} 
      	double discriminate(const CaloTauRef& theCaloTauRef);
	void beginEvent(const edm::Event&, const edm::EventSetup&);

    private:
	TCTauAlgorithm*  tcTauAlgorithm;
};

void TCRecoTauDiscriminationAlgoComponent::beginEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup){
	tcTauAlgorithm->eventSetup(iEvent,iSetup);
}


double TCRecoTauDiscriminationAlgoComponent::discriminate(const CaloTauRef& theCaloTauRef){
	tcTauAlgorithm->recalculateEnergy(*theCaloTauRef);
	return (tcTauAlgorithm->algoComponent());
}

DEFINE_FWK_MODULE(TCRecoTauDiscriminationAlgoComponent);
