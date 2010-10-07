/* 
 * class CaloRecoTauDiscriminationAgainstHadronicJets
 * created : April 21 2010,
 * revised : ,
 * Authors : Sami Lehti (HIP)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/TCTauAlgorithm.h"

class CaloRecoTauDiscriminationAgainstHadronicJets : public CaloTauDiscriminationProducerBase {
    public:
      	explicit CaloRecoTauDiscriminationAgainstHadronicJets(const ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig){   
		tcTauAlgorithm = new TCTauAlgorithm(iConfig);
      	}
      	~CaloRecoTauDiscriminationAgainstHadronicJets(){} 
      	double discriminate(const CaloTauRef& theCaloTauRef);
	void beginEvent(const Event&, const EventSetup&);

    private:
	TCTauAlgorithm*  tcTauAlgorithm;
};

void CaloRecoTauDiscriminationAgainstHadronicJets::beginEvent(const Event& iEvent, const EventSetup& iSetup){
	tcTauAlgorithm->eventSetup(iEvent,iSetup);
}


double CaloRecoTauDiscriminationAgainstHadronicJets::discriminate(const CaloTauRef& theCaloTauRef){
	math::XYZTLorentzVector p4 = tcTauAlgorithm->recalculateEnergy(*theCaloTauRef);
	return ((tcTauAlgorithm->algoComponent() != TCTauAlgorithm::TCAlgoHadronicJet) ? 1. : 0.);
}

DEFINE_FWK_MODULE(CaloRecoTauDiscriminationAgainstHadronicJets);
