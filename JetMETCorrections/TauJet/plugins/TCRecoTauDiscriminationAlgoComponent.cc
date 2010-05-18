/* 
 * class TCRecoTauDiscriminationAlgoComponent
 * created : May 4 2010,
 * revised : ,
 * Authors : Sami Lehti (HIP)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/TCTauAlgorithm.h"

class TCRecoTauDiscriminationAlgoComponent : public CaloTauDiscriminationProducerBase {
    public:
      	explicit TCRecoTauDiscriminationAlgoComponent(const ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig){   
		tcTauAlgorithm = new TCTauAlgorithm(iConfig);
      	}
      	~TCRecoTauDiscriminationAlgoComponent(){} 
      	double discriminate(const CaloTauRef& theCaloTauRef);
	void beginEvent(const Event&, const EventSetup&);

    private:
	TCTauAlgorithm*  tcTauAlgorithm;
};

void TCRecoTauDiscriminationAlgoComponent::beginEvent(const Event& iEvent, const EventSetup& iSetup){
	tcTauAlgorithm->eventSetup(iEvent,iSetup);
}


double TCRecoTauDiscriminationAlgoComponent::discriminate(const CaloTauRef& theCaloTauRef){
	math::XYZTLorentzVector p4 = tcTauAlgorithm->recalculateEnergy(*theCaloTauRef);
	return (tcTauAlgorithm->algoComponent());
}

DEFINE_FWK_MODULE(TCRecoTauDiscriminationAlgoComponent);
