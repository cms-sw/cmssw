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
  
  unsigned nFoundVertex = 0;
  size_t  iVertex = 0;
  unsigned index=0;
  typedef reco::VertexCollection::const_iterator IV;
  for(IV iv=vertices->begin(); iv!=vertices->end(); ++iv, ++index) {
//     cout<<(*iv).x()<<" "
// 	<<(*iv).y()<<" "
// 	<<(*iv).z()<<endl;

    const reco::Vertex& vtx = *iv;
    
    typedef reco::Vertex::trackRef_iterator IT;
    
    // loop on tracks in vertices
    for(IT iTrack=vtx.tracks_begin(); 
	iTrack!=vtx.tracks_end(); ++iTrack) {
	 
      const reco::TrackBaseRef& baseRef = *iTrack;

      // one of the tracks in the vertex is the same as 
      // the track considered in the function
      if(baseRef == trackBaseRef ) {
	iVertex = index;
	nFoundVertex++;
      }	 	
    }
  } 

  if( nFoundVertex == 1) {
    return VertexRef( vertices, iVertex);
  }
  else if(nFoundVertex>1) assert(false);

  assert( !iVertex );

  // no vertex found with this track. 
  // as a secondary solution, associate the closest vertex in z

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
  else 
    return VertexRef();
}


