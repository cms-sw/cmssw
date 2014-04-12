#include "CommonTools/ParticleFlow/interface/PFPileUpAlgo.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

void PFPileUpAlgo::process(const PFCollection & pfCandidates, 
			   const reco::VertexCollection & vertices)  {

  pfCandidatesFromVtx_.clear();
  pfCandidatesFromPU_.clear();

  for( unsigned i=0; i<pfCandidates.size(); i++ ) {
    
    const reco::PFCandidate& cand = * ( pfCandidates[i] );
    
    int ivertex;

    switch( cand.particleId() ) {
    case reco::PFCandidate::h:
      ivertex = chargedHadronVertex( vertices, cand );
      break;
    default:
      continue;
    } 
    
    // no associated vertex, or primary vertex
    // not pile-up
    if( ivertex == -1  || 
	ivertex == 0 ) {
      if(verbose_)
	std::cout<<"VTX "<<i<<" "<< *(pfCandidates[i])<<std::endl;
      pfCandidatesFromVtx_.push_back( pfCandidates[i] );
    } else {
      if(verbose_)
	std::cout<<"PU  "<<i<<" "<< *(pfCandidates[i])<<std::endl;
      // associated to a vertex
      pfCandidatesFromPU_.push_back( pfCandidates[i] );
    }
  }
}


int 
PFPileUpAlgo::chargedHadronVertex( const reco::VertexCollection& vertices, const reco::PFCandidate& pfcand ) const {

  
  reco::TrackBaseRef trackBaseRef( pfcand.trackRef() );
  
  size_t  iVertex = 0;
  unsigned index=0;
  unsigned nFoundVertex = 0;
  typedef reco::VertexCollection::const_iterator IV;
  typedef reco::Vertex::trackRef_iterator IT;
  float bestweight=0;
  for(IV iv=vertices.begin(); iv!=vertices.end(); ++iv, ++index) {

    const reco::Vertex& vtx = *iv;
    
    // loop on tracks in vertices
    for(IT iTrack=vtx.tracks_begin(); 
	iTrack!=vtx.tracks_end(); ++iTrack) {
	 
      const reco::TrackBaseRef& baseRef = *iTrack;

      // one of the tracks in the vertex is the same as 
      // the track considered in the function
      if(baseRef == trackBaseRef ) {
	float w = vtx.trackWeight(baseRef);
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
    return iVertex;
  }
  // no vertex found with this track. 

  // optional: as a secondary solution, associate the closest vertex in z
  if ( checkClosestZVertex_ ) {

    double dzmin = 10000;
    double ztrack = pfcand.vertex().z();
    bool foundVertex = false;
    index = 0;
    for(IV iv=vertices.begin(); iv!=vertices.end(); ++iv, ++index) {

      double dz = fabs(ztrack - iv->z());
      if(dz<dzmin) {
	dzmin = dz; 
	iVertex = index;
	foundVertex = true;
      }
    }

    if( foundVertex ) 
      return iVertex;  

  }


  return -1 ;
}

