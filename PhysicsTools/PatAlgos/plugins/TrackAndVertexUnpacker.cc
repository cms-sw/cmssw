/**
  \class 
  \brief
  \author   Andrea Rizzi
  \version  $Id: TrackAndVertexUnpacker.cc,v 1.2 2010/02/20 21:00:29 wmtan Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//#include "DataFormats/Common/interface/ValueMap.h"
//#include "DataFormats/Common/interface/View.h"
//#include "DataFormats/PatCandidates/interface/Vertexing.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

//#include "PhysicsTools/PatAlgos/interface/VertexingHelper.h"


namespace pat {

  class PATTrackAndVertexUnpacker : public edm::EDProducer {


    public:

      explicit PATTrackAndVertexUnpacker(const edm::ParameterSet & iConfig);
      ~PATTrackAndVertexUnpacker();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

    private:
      typedef std::vector<edm::InputTag> VInputTag;
      // configurables
      edm::EDGetTokenT< std::vector<pat::PackedCandidate> >    Cands_;
      edm::EDGetTokenT<reco::VertexCollection>         PVs_;
      edm::EDGetTokenT<reco::VertexCompositePtrCandidateCollection>         SVs_;
      edm::EDGetTokenT<std::vector<pat::PackedCandidate> >         AdditionalTracks_;
//////    std::vector<edm::EDGetTokenT<edm::View<reco::Candidate> > > particlesTokens_;

  };

}

using pat::PATTrackAndVertexUnpacker;

PATTrackAndVertexUnpacker::PATTrackAndVertexUnpacker(const edm::ParameterSet& iConfig) :
  Cands_(consumes< std::vector<pat::PackedCandidate> >(iConfig.getParameter<edm::InputTag>("packedCandidates"))),
  PVs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("slimmedVertices"))),
  SVs_(consumes<reco::VertexCompositePtrCandidateCollection>(iConfig.getParameter<edm::InputTag>("slimmedSecondaryVertices"))),
  AdditionalTracks_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("additionalTracks")))
{
    produces<reco::TrackCollection>();
    produces<reco::VertexCollection>();
    produces<reco::VertexCollection>("secondary");
}


PATTrackAndVertexUnpacker::~PATTrackAndVertexUnpacker() {
}


void PATTrackAndVertexUnpacker::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
	using namespace edm; using namespace std; using namespace reco;
	Handle<std::vector<pat::PackedCandidate> > cands;
	iEvent.getByToken(Cands_, cands);
	Handle<VertexCollection> pvs;
	iEvent.getByToken(PVs_, pvs);
	Handle<VertexCompositePtrCandidateCollection> svs;
	iEvent.getByToken(SVs_, svs);
	Handle<std::vector<pat::PackedCandidate> > addTracks;
	iEvent.getByToken(AdditionalTracks_, addTracks);

	std::auto_ptr< std::vector<reco::Track> > outTks( new std::vector<reco::Track> );
	std::vector<unsigned int> asso;
	std::map<unsigned int, unsigned int> trackKeys;
	unsigned int j=0;
	for(unsigned int i=0;i<cands->size();i++)	{
		const pat::PackedCandidate & c = (*cands)[i];
		if(c.charge() != 0 && c.numberOfHits()> 0){
			outTks->push_back(c.pseudoTrack());
			if(c.fromPV()==pat::PackedCandidate::PVUsedInFit)
			{
				asso.push_back(j);
			}
 			trackKeys[i]=j;
			j++;
		}	
	}
	reco::Vertex  pv = (*pvs)[0];
	std::auto_ptr< std::vector<reco::Vertex> > outPv( new std::vector<reco::Vertex> );
	int offsetAdd=j;
	for(unsigned int i = 0; i < addTracks->size(); i++) {
	      outTks->push_back((*addTracks)[i].pseudoTrack());
              if((*addTracks)[i].fromPV()==pat::PackedCandidate::PVUsedInFit)
                        {
//				std::cout << "USEDINFIT " << i <<std::endl;
                                asso.push_back(j);
                        }
		 j++;

	}
	edm::OrphanHandle< std::vector<reco::Track>  > oh = iEvent.put( outTks );
	for(unsigned int i=0;i<asso.size();i++)
	{
		TrackRef r(oh,asso[i]);
		TrackBaseRef rr(r);
		pv.add(rr);
	}
	outPv->push_back(pv);
	iEvent.put(outPv);

        std::auto_ptr< std::vector<reco::Vertex> > outSv( new std::vector<reco::Vertex> );
	for(size_t i=0;i< svs->size(); i++) {
		const reco::VertexCompositePtrCandidate &sv = (*svs)[i];	
		outSv->push_back(reco::Vertex(sv.vertex(),sv.vertexCovariance(),sv.vertexChi2(),sv.vertexNdof(),0));
		for(size_t j=0;j<sv.numberOfDaughters();j++){
	                TrackRef r;
			if(sv.daughterPtr(j).id() == cands.id()) {
	                	 r= TrackRef(oh,trackKeys[sv.daughterPtr(j).key()]); // use trackKeys because cand->track has gaps from neutral
			} else {
//				std::cout << "vertex " << i << " using lost Track " << sv.daughterPtr(j).key()  << "  " << offsetAdd+sv.daughterPtr(j).key() << std::endl;  
                                r=TrackRef(oh,offsetAdd+sv.daughterPtr(j).key());  // use directly the key because addTracks is only charged
			}
        	        TrackBaseRef rr(r);
			outSv->back().add(rr);

		}	
	}   

       iEvent.put(outSv,"secondary");

}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATTrackAndVertexUnpacker);
