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
#include "DataFormats/TrackReco/interface/HelixParameters.h"
#include "DataFormats/BTauReco/interface/TrackTagInfo.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfoFwd.h"


namespace reco { 

struct SortByDescendingTrackPt
{
  bool operator()(const TrackRef trStart, const TrackRef trEnd)
  {
    return trStart->pt() > trEnd->pt();
  }
};



  typedef helix::Vector Vector;


  class IsolatedTauTagInfo{

  public:

    IsolatedTauTagInfo() {}
    explicit IsolatedTauTagInfo( const JetTag & p ) : JetTag(p) {}    
    explicit IsolatedTauTagInfo(float discriminator, JetRef jet, TrackTagInfoRefs tracks, float matchingCone, float signalCone, float isolationCone,  float pt_min_tk, float pt_min_lt ) : discriminator_(discriminator), jet_(jet), selectedTracksWTI_(tracks), matchingConeSize_(matchingCone), signalConeSize_(signalCone), isolationConeSize_(isolationCone), pt_min_signal_(pt_min_tk), pt_min_isolation_(pt_min_tk), pt_min_lt_(pt_min_lt)   {}
    
   TrackRefs selectedTracks() const { 
      TrackRefs tracks; 
      for (trackTagInfo_iterator it = selectedTracksWTI_.begin() ; 
	   it != selectedTracksWTI_.end() ; it++) 
	tracks.push_back((*it)->track()); 
      return tracks;
      
}
    
  virtual float discriminator() const { return discriminator_; }

  virtual IsolatedTauTagInfo* clone() const { return new IsolatedTauTagInfo( *this ); }

  

  // matchingConeSize_  centred on the jet direction
  TrackRef leadingSignalTrack(float rm_cone=matchingConeSize_, float pt_min = pt_min_lt_);
    
  // return all tracks in a cone of size "size" around a direction "direction" 
  TrackRefs tracksInCone( const Vector direction, const float size, const float pt_min );

    // return all tracks in a ring between inner radius "inner" and 
    // outer radius "outer" around a direction "direction" 
    TrackRefs tracksInRing( const Vector direction, const float inner, const float outer, const float pt_min);

    //TrackRefs tracksInMatchingCone();
     
    /*   default definition of signal tracks: 
         looks for the highest pT track in a matching cone 
         of size matchingConeSize_ around the jet direction
	 return all tracks within a cone of size signalConeSize_ 
	 around the leading track direction
    */ 
    TrackRefs signalTracks(float signalSize = signalConeSize_, float pt_min = pt_min_signal_) { 
      const Vector direction = leadingSignalTrack()->momentum();
      float size = signalConeSize_;
      float pt_min = pt_min_signal_;
      return tracksInCone( direction, size, pt_min);
    }
   
    /*    default definition of tracks in isolation ring:
	  looks for the highest pT track in a matching cone 
	  of size matchingConeSize_ around the jet direction
	  returns all tracks in a ring between inner radius signalConeSize_
	  and outer radius reconstructionConeSize_ 
    */
    TrackRefs tracksInIsolationRing( float innerRadius = signalConeSize_, float outer Radius = isolationConeSize_, float pt_min = pt_min_isolation_) {
      const Vector direction = leadingSignalTrack()->momentum();
      float innerRadius = signalConeSize_;
      float outerRadius = isolationConeSize_;
      float pt_min = pt_min_isolation__;
      return tracksInRing( direction, innerRadius, outerRadius, pt_min );
    }

    
   

  private:
    
    edm::Ref<JetTagCollection> m_jetTag;

    float matchingConeSize_;
    float signalConeSize_;
    float isolationConeSize_;
    //    float reconstructionConeSize_;
    float pt_min_signal_;
    float pt_min_isolation_;
    float pt_min_lt_;
    float discriminator_;    
   
    //typedef edm::ExtCollection< IsolatedTauTagInfo,JetTagCollection> IsolatedTauExtCollection;
    //typedef edm::OneToOneAssociation<JetTagCollection, IsolatedTauTagInfo> IsolatedTauExtCollection;
  };
}

#endif
