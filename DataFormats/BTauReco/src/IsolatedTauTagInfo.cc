#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <Math/GenVector/VectorUtil.h>

using namespace reco;




TrackRefs IsolatedTauTagInfo::tracksInCone( const Vector direction, const float size, const float pt_min) { 
       
  TrackRefs sTracks;
  for (trackTagInfo_iterator it = selectedTracksWTI_.begin() ; 
	   it != selectedTracksWTI_.end() ; it++) 
    {
      const Track & track  = *(it)->track();
      Vector trackVec = track.momentum();
      float pt_tk = track.pt();
      // Vector

      float deltaR = ROOT::Math::VectorUtil::DeltaR(direction, trackVec);
      if ( deltaR < size && pt_tk > pt_min) sTracks.push_back( &track);
    }
  return sTracks;
}

TrackRefs IsolatedTauTagInfo::tracksInRing( const Vector direction, const float inner, const float outer) { 
       
  TrackRefs sTracks;
  
  for (trackTagInfo_iterator it = selectedTracksWTI_.begin() ; 
	   it != selectedTracksWTI_.end() ; it++) 
    {
      const Track & track  = *(it)->track();
      Vector trackVec = track.momentum();
      float pt_tk = track.pt();
      float deltaR = ROOT::Math::VectorUtil::DeltaR(trackVec,direction);
      
      if ( deltaR > inner && deltaR < outer && pt_tk > pt_min ) sTracks.push_back( &track );
    }
  return sTracks;
}


TrackRef IsolatedTauTagInfo::leadingSignalTrack( float rm_cone=matchingConeSize_, flota pt_min = pt_min_lt_) {
  
  const Jet & j = jet();  
  Vector jetVec( j.px(), j.py(), j.pz() );
  TrackRefs sTracks = tracksInCone(jetVec, rm_cone, pt_min);

  if (sTracks.size() == 0) return NULL;
  
  std::sort( sTracks.begin(), sTracks.end(), SortByDescendingTrackPt() );
  
  if(*sTracks.begin().pt() > pt_min) {
    return *sTracks.begin();
  }else{
    return NULL;
  }

}
