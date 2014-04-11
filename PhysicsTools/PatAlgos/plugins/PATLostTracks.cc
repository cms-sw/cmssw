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
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/Association.h"


namespace pat {
    class PATLostTracks : public edm::EDProducer {
        public:
            explicit PATLostTracks(const edm::ParameterSet&);
            ~PATLostTracks();

            virtual void produce(edm::Event&, const edm::EventSetup&);

        private:
            edm::EDGetTokenT<reco::PFCandidateCollection>    Cands_;
            edm::EDGetTokenT<reco::TrackCollection>         Tracks_;
            edm::EDGetTokenT<reco::VertexCollection>         Vertices_;
            edm::EDGetTokenT<reco::VertexCollection>         PV_;
            double minPt_;
            double minHits_;
            double minPixelHits_;
    };
}

pat::PATLostTracks::PATLostTracks(const edm::ParameterSet& iConfig) :
  Cands_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("inputCandidates"))),
  Tracks_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTracks"))),
  Vertices_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("secondaryVertices"))),
  PV_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertices"))),
  minPt_(iConfig.getParameter<double>("minPt")),
  minHits_(iConfig.getParameter<uint32_t>("minHits")),
  minPixelHits_(iConfig.getParameter<uint32_t>("minPixelHits"))
{
  produces< std::vector<reco::Track> > ();
  produces< std::vector<pat::PackedCandidate> > ();
  produces< edm::Association<pat::PackedCandidateCollection> > ();

}

pat::PATLostTracks::~PATLostTracks() {}

void pat::PATLostTracks::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    edm::Handle<reco::PFCandidateCollection> cands;
    iEvent.getByToken( Cands_, cands );
    std::vector<reco::Candidate>::const_iterator cand;

    edm::Handle<reco::TrackCollection> tracks;
    iEvent.getByToken( Tracks_, tracks );
    
    edm::Handle<reco::VertexCollection> vertices;
    iEvent.getByToken( Vertices_, vertices );

    edm::Handle<reco::VertexCollection> pvs;
    iEvent.getByToken( PV_, pvs );
    reco::VertexRef PV(pvs.id());
    if (!pvs->empty()) {
        PV = reco::VertexRef(pvs, 0);
    }

    std::auto_ptr< std::vector<reco::Track> > outPtrP( new std::vector<reco::Track> );
    std::vector<int> used(tracks->size(),0);

    std::auto_ptr< std::vector<pat::PackedCandidate> > outPtrC( new std::vector<pat::PackedCandidate> );

    //Mark all tracks used in candidates	
    for(unsigned int ic=0, nc = cands->size(); ic < nc; ++ic) {
        const reco::PFCandidate &cand=(*cands)[ic];
        if (cand.charge()) {
	    if(cand.trackRef().isNonnull() && cand.trackRef().id() ==tracks.id())
	    {
		used[cand.trackRef().key()]=1;
            }
	}
    }

    //Mark all tracks used in secondary vertices
    for(unsigned int i=0; i < vertices->size(); i++){
 	const reco::Vertex & sv = (*vertices)[i]; 
	for(reco::Vertex::trackRef_iterator it = sv.tracks_begin(),e=sv.tracks_end(); it!=e;it++){
	    if(used[it->key()]==0)  used[it->key()]=2; // mark as white list 
	}
    }

    std::vector<int> mapping(tracks->size(),-1);
    int j=0;
    for(unsigned int i=0; i < used.size(); i++)
    {
	const reco::Track & tr = (*tracks)[i];
	if(used[i] == 2 || 
	  (used[i]==0 && tr.pt() > minPt_ && tr.numberOfValidHits() >= minHits_ && tr.hitPattern().numberOfValidPixelHits() >= minPixelHits_ ) 
          )
		{
			outPtrP->push_back(tr);
			reco::Candidate::PolarLorentzVector p4(tr.pt(),tr.eta(),tr.phi(),0.13957018);
			outPtrC->push_back(pat::PackedCandidate(p4,tr.vertex(),tr.phi(),0,PV));
			outPtrC->back().setTrackProperties((*tracks)[i]);
			mapping[i]=j;
			j++;
		}
    } 
    iEvent.put(outPtrP);
    edm::OrphanHandle<pat::PackedCandidateCollection> oh =   iEvent.put(outPtrC);
    std::auto_ptr<edm::Association<pat::PackedCandidateCollection> > tk2pc(new edm::Association<pat::PackedCandidateCollection>(oh   ));
    edm::Association<pat::PackedCandidateCollection>::Filler tk2pcFiller(*tk2pc);
    tk2pcFiller.insert(tracks, mapping.begin(), mapping.end());
    tk2pcFiller.fill() ; 
    iEvent.put(tk2pc);

}


using pat::PATLostTracks;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATLostTracks);
