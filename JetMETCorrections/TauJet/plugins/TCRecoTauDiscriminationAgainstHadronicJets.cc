/* 
 * class TCRecoTauDiscriminationAgainstHadronicJets
 * created : April 21 2010,
 * revised : ,
 * Authors : Sami Lehti (HIP)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "JetMETCorrections/TauJet/interface/TCTauCorrector.h"

class TCRecoTauDiscriminationAgainstHadronicJets : public CaloTauDiscriminationProducerBase {
    public:
      	explicit TCRecoTauDiscriminationAgainstHadronicJets(const ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig){   
		tcTauCorrector = new TCTauCorrector(iConfig);
      	}
      	~TCRecoTauDiscriminationAgainstHadronicJets(){} 
      	double discriminate(const CaloTauRef& theCaloTauRef);
	void beginEvent(const Event&, const EventSetup&);

    private:
	TCTauCorrector*  tcTauCorrector;
};

void TCRecoTauDiscriminationAgainstHadronicJets::beginEvent(const Event& iEvent, const EventSetup& iSetup){
	tcTauCorrector->eventSetup(iEvent,iSetup);
}


double TCRecoTauDiscriminationAgainstHadronicJets::discriminate(const CaloTauRef& theCaloTauRef){
	math::XYZTLorentzVector p4 = tcTauCorrector->correctedP4(*theCaloTauRef);
	return ((tcTauCorrector->algoComponent() != TCTauAlgorithm::TCAlgoHadronicJet) ? 1. : 0.);
}

DEFINE_FWK_MODULE(TCRecoTauDiscriminationAgainstHadronicJets);
