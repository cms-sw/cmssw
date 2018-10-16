#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

/* class PFRecoTauDiscriminationByNProngs
 * created : August 30 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 * based on H+ tau ID by Lauri Wendland
 * Modified April 16 2014 by S.Lehti
 */

using namespace reco;
using namespace std;
using namespace edm;

class PFRecoTauDiscriminationByNProngs : public PFTauDiscriminationProducerBase  {
    public:
	explicit PFRecoTauDiscriminationByNProngs(const ParameterSet&);
      	~PFRecoTauDiscriminationByNProngs() override{}

	void beginEvent(const edm::Event&, const edm::EventSetup&) override;
	double discriminate(const reco::PFTauRef&) const override;

    private:
	std::unique_ptr<tau::RecoTauQualityCuts> qcuts_;
	std::unique_ptr<tau::RecoTauVertexAssociator> vertexAssociator_;

	uint32_t minN,maxN;
	bool booleanOutput;
	edm::ParameterSet qualityCuts;
};

PFRecoTauDiscriminationByNProngs::PFRecoTauDiscriminationByNProngs(const ParameterSet& iConfig):
  PFTauDiscriminationProducerBase(iConfig),
  qualityCuts(iConfig.getParameterSet("qualityCuts"))
{
        minN          = iConfig.getParameter<uint32_t>("MinN");
	maxN          = iConfig.getParameter<uint32_t>("MaxN");
        booleanOutput = iConfig.getParameter<bool>("BooleanOutput");

	qcuts_.reset(new tau::RecoTauQualityCuts(qualityCuts.getParameterSet("signalQualityCuts")));
	vertexAssociator_.reset(new tau::RecoTauVertexAssociator(qualityCuts,consumesCollector()));
}

void PFRecoTauDiscriminationByNProngs::beginEvent(const Event& iEvent, const EventSetup& iSetup){
	vertexAssociator_->setEvent(iEvent);
}

double PFRecoTauDiscriminationByNProngs::discriminate(const PFTauRef& tau) const{

	reco::VertexRef pv = vertexAssociator_->associatedVertex(*tau);
	const PFCandidatePtr leadingTrack = tau->leadPFChargedHadrCand();

	uint np = 0;
	if(leadingTrack.isNonnull() && pv.isNonnull()){
	    qcuts_->setPV(pv);
	    qcuts_->setLeadTrack(tau->leadPFChargedHadrCand());

	    for(auto const& cand : tau->signalPFChargedHadrCands() ) {
	        if ( qcuts_->filterCandRef(cand) ) np++;
 	    }
	}

	bool accepted = false;
	if(maxN == 0){
	    if(np == 1 || np == 3) accepted = true;
	}else{
	    if(np >= minN && np <= maxN) accepted = true;
	}

	if(!accepted) np = 0;
	if(booleanOutput) return accepted;
	return np;
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByNProngs);
