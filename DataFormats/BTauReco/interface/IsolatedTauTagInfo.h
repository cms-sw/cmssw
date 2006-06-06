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

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/JetTagFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
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
  bool operator()(const TrackRef trStart, const TrackRef trEnd)
  {
    return trStart->pt() > trEnd->pt();
  }
};



  class IsolatedTauTagInfo{

  public:
    IsolatedTauTagInfo() {}
    IsolatedTauTagInfo(float discriminator, edm::Ref<JetTagCollection> jet, edm::RefVector<TrackCollection> tracks, float matchingCone, float signalCone, float isolationCone,  float pt_min_tk, float pt_min_lt ) : discriminator_(discriminator), m_jetTag(jet), matchingConeSize_(matchingCone), signalConeSize_(signalCone), isolationConeSize_(isolationCone), pt_min_signal_(pt_min_tk), pt_min_isolation_(pt_min_tk), pt_min_lt_(pt_min_lt)   { 
      edm::RefVector<TrackCollection>::const_iterator it = tracks.begin();
      for(;it!= tracks.end(); it++)
	{
	  selectedTracks_.push_back(*it);
	}
    }
    
    ~IsolatedTauTagInfo() {};
    const edm::RefVector<TrackCollection>   selectedTracks() const {return selectedTracks_;}
    
    const edm::Ref<JetTagCollection>  jet() const { return m_jetTag; }
  
  virtual float discriminator() const { return discriminator_; }
  
  virtual IsolatedTauTagInfo* clone() const { return new IsolatedTauTagInfo( *this ); }
  // return all tracks in a cone of size "size" around a direction "direction" 
  //  edm::RefVector<TrackCollection> tracksInCone( edm::Ref<JetTagCollection> myTagJet,  const float size,  const float pt_min );

 /*    
  // matchingConeSize_  centred on the jet direction
    edm::Ref<TrackCollection> leadingSignalTrack( const edm::Ref<JetTagCollection> myTagJet, const float rm_cone, const float pt_min);

 
    // return all tracks in a ring between inner radius "inner" and 
    // outer radius "outer" around a direction "direction" 
    TrackRefs tracksInRing( const Vector direction, const float inner, const float outer, const float pt_min);

    //TrackRefs tracksInMatchingCone();

    TrackRefs signalTracks(float signalSize = signalConeSize_, float pt_min = pt_min_signal_) { 
      const Vector direction = leadingSignalTrack()->momentum();
      float size = signalConeSize_;
      float pt_min = pt_min_signal_;
      return tracksInCone( direction, size, pt_min);
    }
   

    TrackRefs tracksInIsolationRing( float innerRadius = signalConeSize_, float outer Radius = isolationConeSize_, float pt_min = pt_min_isolation_) {
      const Vector direction = leadingSignalTrack()->momentum();
      float innerRadius = signalConeSize_;
      float outerRadius = isolationConeSize_;
      float pt_min = pt_min_isolation__;
      return tracksInRing( direction, innerRadius, outerRadius, pt_min );
    }

    */

  private:
    


    float matchingConeSize_;
    float signalConeSize_;
    float isolationConeSize_;
    //    float reconstructionConeSize_;
    float pt_min_signal_;
    float pt_min_isolation_;
    float pt_min_lt_;
    float discriminator_;    
    edm::Ref<JetTagCollection> m_jetTag;
    edm::RefVector<TrackCollection> selectedTracks_;
    JetTracksAssociationRef m_jetTracksAssociation;


    //typedef edm::ExtCollection< IsolatedTauTagInfo,JetTagCollection> IsolatedTauExtCollection;
    //typedef edm::OneToOneAssociation<JetTagCollection, IsolatedTauTagInfo> IsolatedTauExtCollection;
  };
}

#endif
