/* 
 * class TCRecoTauDiscriminationAgainstHadronicJets
 * created : April 21 2010,
 * revised : ,
 * Authors : Sami Lehti (HIP)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/TCTauAlgorithm.h"

using namespace reco;

class TCRecoTauDiscriminationAgainstHadronicJets : public CaloTauDiscriminationProducerBase {
    public:
      	explicit TCRecoTauDiscriminationAgainstHadronicJets(const edm::ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig){   
		tcTauAlgorithm = new TCTauAlgorithm(iConfig);
      	}
      	~TCRecoTauDiscriminationAgainstHadronicJets(){} 
      	double discriminate(const CaloTauRef& theCaloTauRef);
	void beginEvent(const edm::Event&, const edm::EventSetup&);

    private:
	TCTauAlgorithm*  tcTauAlgorithm;
};

void TCRecoTauDiscriminationAgainstHadronicJets::beginEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup){
	tcTauAlgorithm->eventSetup(iEvent,iSetup);
}


double TCRecoTauDiscriminationAgainstHadronicJets::discriminate(const CaloTauRef& theCaloTauRef){
	math::XYZTLorentzVector p4 = tcTauAlgorithm->recalculateEnergy(*theCaloTauRef);
	return ((tcTauAlgorithm->algoComponent() != TCTauAlgorithm::TCAlgoHadronicJet) ? 1. : 0.);
}

DEFINE_FWK_MODULE(TCRecoTauDiscriminationAgainstHadronicJets);
