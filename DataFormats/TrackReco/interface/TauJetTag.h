#ifndef BTauReco_TauJetTag_h
#define BTauReco_TauJetTag_h
//
// \class TauJetTag
// \short concrete class for the description of the result of the tau-tagging algorithm 
//
// concrete class inherits from JetTag
// contains the result of the tracker based tau-tagging algorithm 
// object to be made persistent on RECO
//
// \author: Marcel Vos, based on ORCA class by S. Gennai and F. Moortgat
//

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/HelixParameters.h"


namespace reco { 

struct SortByDescendingTrackPt
{
  bool operator()(const Track* trStart, const Track* trEnd)
  {
    return trStart->pt() > trEnd->pt();
  }
};



  typedef helix::Vector Vector;
  typedef std::vector<int>::size_type size_type; /// MVS need this to iterate


  class TauJetTag: public JetTag {

  public:


    explicit TauJetTag() {}
    explicit TauJetTag( const JetTag & p ) : JetTag(p) {}    
    TauJetTag(float dRmatch, 
	      float dRsignal, 
	      float dRreco,
	      float discriminator, 
	      JetRef jet, 
	      TrackRefs tracks) : 
      matchingConeSize_ (dRmatch), 
      signalConeSize_ (dRsignal), 
      reconstructionConeSize_ (dRreco), 
      discriminator_ (discriminator), 
      jet_(jet), 
      selectedTracks_ (tracks)  {}


    virtual float discriminator() const { return discriminator_; }

    virtual JetRef jetRef() const { return jet_; }

    virtual const Jet & jet() const { return *jet_; }

    virtual TrackRefs selectedTracks() const { return selectedTracks_;}

    virtual TauJetTag* clone() const { return new TauJetTag( *this ); }

    bool isIsolated() {
      if (tracksInIsolationRing().size()) return false;
      else return true;
    }

    // highest pT track reconstructed within a cone of radius 
    // matchingConeSize_  centred on the jet direction
    const Track* leadingSignalTrack();
    
    // return all tracks in a cone of size "size" around a direction "direction" 
    std::vector<const Track*> tracksInCone( const Vector direction, const float size );

    // return all tracks in a ring between inner radius "inner" and 
    // outer radius "outer" around a direction "direction" 
    std::vector<const Track*> tracksInRing( const Vector direction, const float inner, const float outer);

    std::vector<const Track*> tracksInMatchingCone();
     
    /*   default definition of signal tracks: 
         looks for the highest pT track in a matching cone 
         of size matchingConeSize_ around the jet direction
	 return all tracks within a cone of size signalConeSize_ 
	 around the leading track direction
    */ 
    std::vector<const Track*> signalTracks() { 
      const Vector direction = leadingSignalTrack()->momentum();
      float size = signalConeSize_;
      return tracksInCone( direction, size );
    }
   
    /*    default definition of tracks in isolation ring:
	  looks for the highest pT track in a matching cone 
	  of size matchingConeSize_ around the jet direction
	  returns all tracks in a ring between inner radius signalConeSize_
	  and outer radius reconstructionConeSize_ 
    */
    std::vector<const Track*> tracksInIsolationRing() {
      const Vector direction = leadingSignalTrack()->momentum();
      float innerRadius = signalConeSize_;
      float outerRadius = reconstructionConeSize_;
      return tracksInRing( direction, innerRadius, outerRadius );
    }

    
  

  private:
    


    float matchingConeSize_;
    float signalConeSize_;
    float reconstructionConeSize_;

    float discriminator_;    
    JetRef jet_;
    TrackRefs selectedTracks_;
		    
    
  };
}

#endif
