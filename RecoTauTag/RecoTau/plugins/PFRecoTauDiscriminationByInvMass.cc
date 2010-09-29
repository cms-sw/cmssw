#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

/* class PFRecoTauDiscriminationByInvMass
 * created : August 30 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 * based on H+ tau ID by Lauri Wendland
 */

#include "TLorentzVector.h"

using namespace reco;
using namespace std;

class PFRecoTauDiscriminationByInvMass : public PFTauDiscriminationProducerBase  {
    public:
	explicit PFRecoTauDiscriminationByInvMass(const ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig) {
		invMassMin		= iConfig.getParameter<double>("invMassMin");
		invMassMax		= iConfig.getParameter<double>("invMassMax");
		chargedPionMass 	= 0.139;
		booleanOutput 		= iConfig.getParameter<bool>("BooleanOutput");
	}

      	~PFRecoTauDiscriminationByInvMass(){}

	void beginEvent(const edm::Event&, const edm::EventSetup&);
	double discriminate(const reco::PFTauRef&);

    private:
	double threeProngInvMass(const PFTauRef&);

	double chargedPionMass;

	double invMassMin,invMassMax;

	bool booleanOutput;
};

void PFRecoTauDiscriminationByInvMass::beginEvent(const Event& iEvent, const EventSetup& iSetup){
}

double PFRecoTauDiscriminationByInvMass::discriminate(const PFTauRef& tau){

	double invMass = threeProngInvMass(tau);
	if(booleanOutput) return ( invMass > invMassMin && invMass < invMassMax ? 1. : 0. );
	return invMass;
}

double PFRecoTauDiscriminationByInvMass::threeProngInvMass(const PFTauRef& tau){
	TLorentzVector sum;
	PFCandidateRefVector signalTracks = tau->signalPFChargedHadrCands();
        for(size_t i = 0; i < signalTracks.size(); ++i){                        
                TLorentzVector p4;
                p4.SetXYZM(signalTracks[i]->px(), 
                           signalTracks[i]->py(),
                           signalTracks[i]->pz(),
                           chargedPionMass);
                sum += p4;
        }
	return sum.M();
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByInvMass);

