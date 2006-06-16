#ifndef BTauReco_TauTagIsolation_h
#define BTauReco_TauTagIsolation_h
//
// \class TauJetTagIsolation
// \short concrete class for the description of the result of the tau-tagging with the isolation algorithm 
//
// concrete class inherits from JetTag
// contains the result of the tracker based tau-tagging algorithm 
// object to be made persistent on RECO
//
// \author: Marcel Vos and Simone Gennai, based on ORCA class by S. Gennai and F. Moortgat
//


#include "DataFormats/BTauReco/interface/JetTagFwd.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfoFwd.h"
//#include "DataFormats/TrackReco/interface/HelixParameters.h"
#include "DataFormats/Math/interface/Vector3D.h"
//Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
//
using namespace std;

namespace reco { 

struct SortByDescendingTrackPt
{
  bool operator()(const TrackRef* trStart, const TrackRef* trEnd)
  {
    return (*trStart)->pt() > (*trEnd)->pt();
  }
};



  class IsolatedTauTagInfo{

  public:
    IsolatedTauTagInfo() {}
    IsolatedTauTagInfo(edm::RefVector<TrackCollection> tracks) 
      {    
	track_iterator it = tracks.begin();
	for(;it!= tracks.end(); it++)
	  {
	    selectedTracks_.push_back(*it);
	  }
	
    }
    
    virtual ~IsolatedTauTagInfo() {};
    
    void setJetTag(const JetTagRef myRef) { 
       m_jetTag = myRef;
     }
    const Jet & jet() const { return m_jetTag->jet(); }
    
    const TrackRefVector & allTracks() const { return (*m_jetTag).tracks(); }


    const TrackRefVector & selectedTracks() const {return selectedTracks_;}
    
    const JetTagRef & jetRef() const { return m_jetTag; }
  
     double discriminator() const { 
       double myDiscr = m_jetTag->discriminator();
       return myDiscr; }
     double discriminator(float m_cone, float sig_cone, float iso_con, float pt_min_lt, float pt_min_tk, int nTracksIsoRing=0) const;
     double discriminator( math::XYZVector myVector, float m_cone, float sig_cone, float iso_con, float pt_min_lt, float pt_min_tk, int nTracksIsoRing=0) const;
    virtual IsolatedTauTagInfo* clone() const { return new IsolatedTauTagInfo( *this ); }
    // return all tracks in a cone of size "size" around a direction "direction" 
    const edm::RefVector<TrackCollection> tracksInCone(const math::XYZVector myVector,const float size,  const float pt_min ) const;
 
    // matchingConeSize_  centred on the jet direction
    const TrackRef leadingSignalTrack(const float rm_cone, const float pt_min) const;
    const TrackRef leadingSignalTrack(math::XYZVector myVector, const float rm_cone, const float pt_min) const;
    
    
  private:
    



    JetTagRef m_jetTag;
    TrackRefVector selectedTracks_;
    TrackRef track;
  };
}

#endif
