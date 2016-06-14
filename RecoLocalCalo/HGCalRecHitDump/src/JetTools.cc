#include "RecoLocalCalo/HGCalRecHitDump/interface/JetTools.h"

//
std::pair<float,float> betaVariables(const reco::PFJet * jet, 	       
				     const reco::Vertex * vtx,
				     const reco::VertexCollection & allvtx)
{
  //iterate over constituents
  float beta(0.), betaStar(0.), sumTkPt(0.);
  std::vector <reco::PFCandidatePtr> constituents = jet->getPFConstituents();
  for(std::vector <reco::PFCandidatePtr>::iterator it=constituents.begin(); 
      it!=constituents.end(); 
      ++it) 
    {
      reco::PFCandidatePtr & icand = *it;
      if( !icand->trackRef().isNonnull() || !icand->trackRef().isAvailable() ) continue;
      float tkpt = icand->trackRef()->pt(); 
      sumTkPt += tkpt;

      // 'classic' beta definition based on track-vertex association
      bool inVtx0 = find( vtx->tracks_begin(), vtx->tracks_end(), reco::TrackBaseRef(icand->trackRef()) ) != vtx->tracks_end();
      bool inAnyOther = false;
      for(reco::VertexCollection::const_iterator  vi=allvtx.begin(); vi!=allvtx.end(); ++vi ) 
	{
	  if(inAnyOther) continue;
	  const reco::Vertex & iv = *vi;
	  if( iv.isFake() || iv.ndof() < 4 ) { continue; }
	  // the primary vertex may have been copied by the user: check identity by position
	  bool isVtx0  = (iv.position() - vtx->position()).r() < 0.02;
	  if(isVtx0) continue;
	  inAnyOther = find( iv.tracks_begin(), iv.tracks_end(), reco::TrackBaseRef(icand->trackRef()) ) != iv.tracks_end();
	}

      // classic beta/betaStar
      if( inVtx0 && ! inAnyOther )       beta     += tkpt;
      else if( ! inVtx0 && inAnyOther )  betaStar += tkpt;
    }

  if(sumTkPt>0)
    {
      beta /= sumTkPt;
      betaStar /= sumTkPt;
    }

  return std::make_pair(beta,betaStar);
}
