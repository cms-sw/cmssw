#include "DataFormats/BTauReco/interface/TauJetTag.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <Math/GenVector/VectorUtil.h>

using namespace reco;




std::vector<const Track*> TauJetTag::tracksInCone( const Vector direction, const float size) { 
       
  std::vector<const Track*> sTracks;
  for ( size_type i = 0 ; i < selectedTracks_.size(); i++ ) 
    {
      const Track & track  = *(selectedTracks_[i]);
      Vector trackVec = track.momentum();
      // Vector

      float deltaR = ROOT::Math::VectorUtil::DeltaR(direction, trackVec);
      if ( deltaR < size ) sTracks.push_back( &track );
    }
  return sTracks;
}

std::vector<const Track*> TauJetTag::tracksInRing( const Vector direction, const float inner, const float outer) { 
       
  std::vector<const Track*> sTracks;
  for ( size_type i = 0 ; i < selectedTracks_.size(); i++ ) 
    {
      const Track & track  = *(selectedTracks_[i]);
      Vector trackVec = track.momentum();
      float deltaR = ROOT::Math::VectorUtil::DeltaR(trackVec,direction);
      
      if ( deltaR > inner && deltaR < outer ) sTracks.push_back( &track );
    }
  return sTracks;
}


std::vector<const Track*> TauJetTag::tracksInMatchingCone()
{ 
  const Jet & j = jet();  
  Vector jetVec( j.getPx(), j.getPy(), j.getPz() );
  return tracksInCone( jetVec, matchingConeSize_ );
}


const Track* TauJetTag::leadingSignalTrack() {
  
  std::vector<const Track*> sTracks = tracksInMatchingCone();

  if (sTracks.size() == 0) return NULL;
  
  std::sort( sTracks.begin(), sTracks.end(), SortByDescendingTrackPt() );

  return *sTracks.begin();

}
