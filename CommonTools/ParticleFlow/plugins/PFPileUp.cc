#include "CommonTools/ParticleFlow/plugins/PFPileUp.h"

#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/ESHandle.h"

// #include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Math/interface/deltaR.h"

using namespace std;
using namespace edm;
using namespace reco;

PFPileUp::PFPileUp(const edm::ParameterSet& iConfig) {
  


  inputTagPFCandidates_ 
    = iConfig.getParameter<InputTag>("PFCandidates");

  inputTagVertices_ 
    = iConfig.getParameter<InputTag>("Vertices");

  enable_ = iConfig.getParameter<bool>("Enable");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);


  if ( iConfig.exists("checkClosestZVertex") ) {
    checkClosestZVertex_ = iConfig.getParameter<bool>("checkClosestZVertex");
  } else {
    checkClosestZVertex_ = false;
  }


  produces<reco::PileUpPFCandidateCollection>();
  
}



PFPileUp::~PFPileUp() { }



void PFPileUp::beginJob() { }


void PFPileUp::produce(Event& iEvent, 
			  const EventSetup& iSetup) {
  
//   LogDebug("PFPileUp")<<"START event: "<<iEvent.id().event()
// 			 <<" in run "<<iEvent.id().run()<<endl;
  
   
  // get PFCandidates

  auto_ptr< reco::PileUpPFCandidateCollection > 
    pOutput( new reco::PileUpPFCandidateCollection ); 
  
  if(enable_) {

    Handle<PFCandidateCollection> pfCandidates;
    iEvent.getByLabel( inputTagPFCandidates_, pfCandidates);

  
    // get vertices 

    Handle<VertexCollection> vertices;
    iEvent.getByLabel( inputTagVertices_, vertices);
  
    for( unsigned i=0; i<pfCandidates->size(); i++ ) {
    
      // const reco::PFCandidate& cand = (*pfCandidates)[i];
      PFCandidatePtr candptr(pfCandidates, i);

      //     PFCandidateRef pfcandref(pfCandidates,i); 

      VertexRef vertexref;

      switch( candptr->particleId() ) {
      case PFCandidate::h:
	vertexref = chargedHadronVertex( vertices, *candptr );
	break;
      default:
	continue;
      } 
    
      // no associated vertex, or primary vertex
      // not pile-up
      if( vertexref.isNull() || 
	  vertexref.key()==0 ) continue;

      pOutput->push_back( PileUpPFCandidate( candptr, vertexref ) );
      pOutput->back().setSourceCandidatePtr( candptr );
    }
  }
  iEvent.put( pOutput );
  
}



VertexRef 
PFPileUp::chargedHadronVertex( const Handle<VertexCollection>& vertices, const PFCandidate& pfcand ) const {

  
  reco::TrackBaseRef trackBaseRef( pfcand.trackRef() );
  
  size_t  iVertex = 0;
  unsigned index=0;
  unsigned nFoundVertex = 0;
  typedef reco::VertexCollection::const_iterator IV;
  float bestweight=0;
  for(IV iv=vertices->begin(); iv!=vertices->end(); ++iv, ++index) {

    const reco::Vertex& vtx = *iv;
    
    typedef reco::Vertex::trackRef_iterator IT;
    
    // loop on tracks in vertices
    for(IT iTrack=vtx.tracks_begin(); 
	iTrack!=vtx.tracks_end(); ++iTrack) {
	 
      const reco::TrackBaseRef& baseRef = *iTrack;

      // one of the tracks in the vertex is the same as 
      // the track considered in the function
      float w = vtx.trackWeight(baseRef);
      if(baseRef == trackBaseRef ) {
	//select the vertex for which the track has the highest weight
	if (w > bestweight){
	  bestweight=w;
	  iVertex=index;
	  nFoundVertex++;
	}	 	
      }
    }
  }

  if (nFoundVertex>0){
    if (nFoundVertex!=1)
      edm::LogWarning("TrackOnTwoVertex")<<"a track is shared by at least two verteces. Used to be an assert";
    return VertexRef( vertices, iVertex);
  }
  // no vertex found with this track. 

  // optional: as a secondary solution, associate the closest vertex in z
  if ( checkClosestZVertex_ ) {

    double dzmin = 10000;
    double ztrack = pfcand.vertex().z();
    bool foundVertex = false;
    index = 0;
    for(IV iv=vertices->begin(); iv!=vertices->end(); ++iv, ++index) {

      double dz = fabs(ztrack - iv->z());
      if(dz<dzmin) {
	dzmin = dz; 
	iVertex = index;
	foundVertex = true;
      }
    }

    if( foundVertex ) 
      return VertexRef( vertices, iVertex);  

  }


  return VertexRef();
}


