#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "FWCore/Utilities/interface/InputTag.h"

/* class CaloRecoTauDiscriminationByDeltaE
 * created : September 23 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 * based on H+ tau ID by Lauri Wendland
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "TLorentzVector.h"

using namespace reco;
using namespace std;
using namespace edm;

class CaloRecoTauDiscriminationByDeltaE : public CaloTauDiscriminationProducerBase  {
    public:
	explicit CaloRecoTauDiscriminationByDeltaE(const ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig){
		deltaEmin		= iConfig.getParameter<double>("deltaEmin");
		deltaEmax               = iConfig.getParameter<double>("deltaEmax");
		chargedPionMass         = 0.139;
		booleanOutput 		= iConfig.getParameter<bool>("BooleanOutput");
	}

      	~CaloRecoTauDiscriminationByDeltaE(){}

	void beginEvent(const edm::Event&, const edm::EventSetup&);
	double discriminate(const reco::CaloTauRef&);

    private:
	double DeltaE(const CaloTauRef&);

	double chargedPionMass;

	double deltaEmin,deltaEmax;
	bool booleanOutput;
};

void CaloRecoTauDiscriminationByDeltaE::beginEvent(const Event& iEvent, const EventSetup& iSetup){
}

double CaloRecoTauDiscriminationByDeltaE::discriminate(const CaloTauRef& tau){

	double dE = DeltaE(tau);
	if(booleanOutput) return ( dE > deltaEmin && dE < deltaEmax ? 1. : 0. );
	return dE;
}

double CaloRecoTauDiscriminationByDeltaE::DeltaE(const CaloTauRef& tau){
	double tracksE = 0;
	reco::TrackRefVector signalTracks = tau->signalTracks();
	for(size_t i = 0; i < signalTracks.size(); ++i){
		TLorentzVector p4;
		p4.SetXYZM(signalTracks[i]->px(),
               signalTracks[i]->py(),
               signalTracks[i]->pz(),
               chargedPionMass);
		tracksE += p4.E();
	}
	if(tau->leadTrackHCAL3x3hitsEtSum() == 0) return -1; // electron
	return tracksE/tau->leadTrackHCAL3x3hitsEtSum() - 1.0;
}

DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByDeltaE);

