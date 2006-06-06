#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <Math/GenVector/VectorUtil.h>
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

using namespace edm;
using namespace reco;
/*
RefVector<TrackCollection> IsolatedTauTagInfo::tracksInCone( Ref<JetTagCollection> myTagJet,  const float size,  const float pt_min) { 
       
  CaloJet myjet = myTagJet->jet();
   RefVector<TrackCollection> tmp = myTagJet->tracks();


  math::XYZVector jet3Vec   (m_jet->px(),m_jet->py(),m_jet->pz()) ;
  RefVector<TrackCollection>::const_iterator myTrack = selectedTracks_.begin();
  for(;myTracks != selectedTracks_.end(); myTracks++)
    {
      math::XYZVector trackMomentum = myTrack->momentum() ;
      float pt_tk = myTrack->momentum.pt();
      float deltaR = ROOT::Math::VectorUtil::DeltaR(jet3Vec, trackMomentum);
      if ( deltaR < size && pt_tk > pt_min) tmp.push_back( *myTrack);
      }

  return tmp;
}
*/
/*
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


TrackRef IsolatedTauTagInfo::leadingSignalTrack(const edm::Ref<JetTagCollection> myTagJet, const float rm_cone, const float pt_min) {


  edm::RefVector<TrackCollection>  sTracks = tracksInCone(myTagJet, rm_cone, pt_min);

  if (sTracks.size() == 0) return NULL;
  
  std::sort( sTracks.begin(), sTracks.end(), SortByDescendingTrackPt() );
  
  if(*sTracks.begin().pt() > pt_min) {
    return *sTracks.begin();
  }else{
    return NULL;
  }

}
*/
