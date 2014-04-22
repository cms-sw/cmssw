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
      edm::EDGetTokenT<reco::TrackCollection>         AdditionalTracks_;
//////    std::vector<edm::EDGetTokenT<edm::View<reco::Candidate> > > particlesTokens_;

  };

}

using pat::PATTrackAndVertexUnpacker;

PATTrackAndVertexUnpacker::PATTrackAndVertexUnpacker(const edm::ParameterSet& iConfig) :
  Cands_(consumes< std::vector<pat::PackedCandidate> >(iConfig.getParameter<edm::InputTag>("packedCandidates"))),
  PVs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("slimmedVertices"))),
  AdditionalTracks_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("additionalTracks")))
{
    produces<reco::TrackCollection>();
    produces<reco::VertexCollection>();
}


PATTrackAndVertexUnpacker::~PATTrackAndVertexUnpacker() {
}


void PATTrackAndVertexUnpacker::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
	using namespace edm; using namespace std; using namespace reco;
	Handle<std::vector<pat::PackedCandidate> > cands;
	iEvent.getByToken(Cands_, cands);
	Handle<VertexCollection> pvs;
	iEvent.getByToken(PVs_, pvs);
	Handle<TrackCollection> addTracks;
	iEvent.getByToken(AdditionalTracks_, addTracks);

	std::auto_ptr< std::vector<reco::Track> > outTks( new std::vector<reco::Track> );
	std::vector<unsigned int> asso;
	unsigned int j=0;
	for(unsigned int i=0;i<cands->size();i++)	{
		const pat::PackedCandidate & c = (*cands)[i];
		if(c.charge() != 0 && c.pt() > 0.95){
			outTks->push_back(c.pseudoTrack());
			if(c.fromPV()==pat::PackedCandidate::PVUsedInFit)
			{
				asso.push_back(j);
			}
			j++;
		}	
	}
	reco::Vertex  pv = (*pvs)[0];
	std::auto_ptr< std::vector<reco::Vertex> > outPv( new std::vector<reco::Vertex> );
	for(unsigned int i = 0; i < addTracks->size(); i++) {
	      outTks->push_back((*addTracks)[i]);
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
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATTrackAndVertexUnpacker);
