#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

/* class PFRecoTauDiscriminationByNProngs
 * created : August 30 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 * based on H+ tau ID by Lauri Wendland
 */

using namespace reco;
using namespace std;
using namespace edm;

class PFRecoTauDiscriminationByNProngs : public PFTauDiscriminationProducerBase  {
    public:
	explicit PFRecoTauDiscriminationByNProngs(const ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig){
		nprongs			= iConfig.getParameter<uint32_t>("nProngs");
		booleanOutput = iConfig.getParameter<bool>("BooleanOutput");
	}

      	~PFRecoTauDiscriminationByNProngs(){}

	void beginEvent(const edm::Event&, const edm::EventSetup&);
	double discriminate(const reco::PFTauRef&);

    private:

	uint32_t nprongs;
	bool booleanOutput;
};

void PFRecoTauDiscriminationByNProngs::beginEvent(const Event& iEvent, const EventSetup& iSetup){}

double PFRecoTauDiscriminationByNProngs::discriminate(const PFTauRef& tau){

	bool accepted = false;
	int np = tau->signalTracks().size();

	if((np == 1 && (nprongs == 1 || nprongs == 0)) ||
           (np == 3 && (nprongs == 3 || nprongs == 0)) ) accepted = true;

	if(!accepted) np = 0;
	if(booleanOutput) return accepted;
	return np;
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByNProngs);

