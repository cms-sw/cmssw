#ifndef BTauReco_BJetTagTrackCounting_h
#define BTauReco_BJetTagTrackCounting_h
 

#include <vector> 

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"


namespace reco {
 
class BJetTagTrackCounting : public JetTag
 {
  public:


   BJetTagTrackCounting() {}
   explicit BJetTagTrackCounting( const JetTag & p ) : JetTag( p ) { }
 
   explicit BJetTagTrackCounting(float discriminator, JetRef jet, TrackRefs tracks) : discriminator_(discriminator), jet_(jet), selectedTracks_(tracks) {}
 


  virtual TrackRefs selectedTracks() const {return selectedTracks_;}

  virtual JetRef jetRef() const { return jet_; }
  virtual const Jet & jet() const { return *jet_; }
  
  virtual float discriminator() const { return discriminator_; }
  
  virtual BJetTagTrackCounting* clone() const { return new BJetTagTrackCounting( * this ); }



  float discriminator(unsigned int i) const { if (i < selectedTracks_.size()) return selectedTracks_[i]; else return 0;}
  

 
  private:
  float discriminator_;
  JetRef jet_;
  TrackRefs selectedTracks_;

    
 };
 
}
#endif
