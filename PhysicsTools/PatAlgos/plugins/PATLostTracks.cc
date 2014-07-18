#include <string>


#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"


namespace pat {
    class PATLostTracks : public edm::EDProducer {
        public:
            explicit PATLostTracks(const edm::ParameterSet&);
            ~PATLostTracks();

            virtual void produce(edm::Event&, const edm::EventSetup&);

        private:
            edm::EDGetTokenT<reco::PFCandidateCollection>    Cands_;
            edm::EDGetTokenT<reco::TrackCollection>         Tracks_;
            double minPt_;
            double minHits_;
            double minPixelHits_;
    };
}

pat::PATLostTracks::PATLostTracks(const edm::ParameterSet& iConfig) :
  Cands_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("inputCandidates"))),
  Tracks_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTracks"))),
  minPt_(iConfig.getParameter<double>("minPt")),
  minHits_(iConfig.getParameter<uint32_t>("minHits")),
  minPixelHits_(iConfig.getParameter<uint32_t>("minPixelHits"))

{
  produces< std::vector<reco::Track> > ();
}

pat::PATLostTracks::~PATLostTracks() {}

void pat::PATLostTracks::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    edm::Handle<reco::PFCandidateCollection> cands;
    iEvent.getByToken( Cands_, cands );
    std::vector<reco::Candidate>::const_iterator cand;

    edm::Handle<reco::TrackCollection> Tracks;
    iEvent.getByToken( Tracks_, Tracks );
    
    std::auto_ptr< std::vector<reco::Track> > outPtrP( new std::vector<reco::Track> );
    std::vector<bool> used(Tracks->size());


    for(unsigned int ic=0, nc = cands->size(); ic < nc; ++ic) {
        const reco::PFCandidate &cand=(*cands)[ic];
        if (cand.charge()) {
	    if(cand.trackRef().isNonnull() && cand.trackRef().id() == Tracks.id())
	    {
		used[cand.trackRef().key()]=true;
            }
	}
    }

    for(unsigned int i=0; i < used.size(); i++)
    {
	if(!used[i] && (*Tracks)[i].pt() > minPt_ && 
	(*Tracks)[i].numberOfValidHits() >= minHits_ &&
	        (*Tracks)[i].hitPattern().numberOfValidPixelHits() >= minPixelHits_ ) outPtrP->push_back((*Tracks)[i]);
    } 

    iEvent.put(outPtrP);

}


using pat::PATLostTracks;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATLostTracks);
