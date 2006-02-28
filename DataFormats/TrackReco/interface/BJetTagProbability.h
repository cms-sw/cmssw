#ifndef BTauReco_BJetTagProbability_h
#define BTauReco_BJetTagProbability_h
//
// \class BJetTagProbability
// \short concrete class for the description of the result of the combined b-tagging algorithm 
//
// concrete class, inherits from JetTag
// contains the result of combined b-tagging algorithm 
// object to be made persistent on RECO
//
// \author: Christian Weiser, Andrea Rizzi, Marcel Vos
//

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TrackWithTagInfo.h"

namespace reco {
  class BJetTagProbability : public JetTag {
 
  //
  
    
  public:

    BJetTagProbability() {}

    explicit BJetTagProbability(const JetTag & p ) : JetTag( p ) { }
    explicit BJetTagProbability(float discriminator, JetRef jet, TrackWithTagInfoRefs tracks) : discriminator_ (discriminator), jet_(jet), selectedTracksWTI_ (tracks)  {}


    virtual float discriminator() const { return discriminator_; }

    virtual JetRef jetRef() const { return jet_; }
    virtual const Jet & jet() const { return *jet_; }

    virtual TrackRefs selectedTracks() const { 
      TrackRefs tracks; 
      for (trackWithTagInfo_iterator it = selectedTracksWTI_.begin() ; 
	                             it != selectedTracksWTI_.end() ; it++) 
	tracks.push_back((*it)->track()); 
      return tracks;
    }
    virtual BJetTagProbability* clone() const { return new BJetTagProbability( *this ); }
    TrackWithTagInfoRefs selectedTracksWithTagInfo() const { return selectedTracksWTI_; }
 
    

    //     float trackProbability( unsigned int i ) const { if (i<selectedTracksWTI_.size()) return (selectedTracksWTI_[i].product()->at(i))->probability(); else return 0;}
    /*
    trackWithTagInfo trackWithTagInfoPUV(unsigned int i) const { 
      return selectedTracksWTI_[i].product()->at(i); 
     
    }
    */
   float getDummy() { return dummy_; }
    void setDummy(float d) { dummy_ = d;}
    
  

  private:
    
      
    float dummy_;
    float discriminator_;
    
    JetRef jet_;
    TrackWithTagInfoRefs selectedTracksWTI_;
		    
    
  };
}

#endif
