/* 
 * class TCRecoTauDiscriminationAlgoComponent
 * created : May 4 2010,
 * revised : ,
 * Authors : Sami Lehti (HIP)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "JetMETCorrections/TauJet/interface/TCTauCorrector.h"

class TCRecoTauDiscriminationAlgoComponent : public CaloTauDiscriminationProducerBase {
    public:
      	explicit TCRecoTauDiscriminationAlgoComponent(const ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig){   
		tcTauCorrector = new TCTauCorrector(iConfig);
      	}
      	~TCRecoTauDiscriminationAlgoComponent(){} 
      	double discriminate(const CaloTauRef& theCaloTauRef);
	void beginEvent(const Event&, const EventSetup&);

    private:
	TCTauCorrector*  tcTauCorrector;
};

void TCRecoTauDiscriminationAlgoComponent::beginEvent(const Event& iEvent, const EventSetup& iSetup){
	tcTauCorrector->eventSetup(iEvent,iSetup);
}


double TCRecoTauDiscriminationAlgoComponent::discriminate(const CaloTauRef& theCaloTauRef){
	math::XYZTLorentzVector p4 = tcTauCorrector->correctedP4(*theCaloTauRef);
	return (tcTauCorrector->algoComponent());
}

DEFINE_FWK_MODULE(TCRecoTauDiscriminationAlgoComponent);
