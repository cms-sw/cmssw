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

namespace reco {
  class TauJetTag: public JetTag {
 
  public:


    explicit TauJetTag() {}
    explicit TauJetTag( const JetTag & p ) : JetTag(p) {}    
    TauJetTag(float discriminator, JetRef jet, TrackRefs tracks) : discriminator_ (discriminator), jet_(jet), selectedTracks_ (tracks)  {}


    virtual float discriminator() const { return discriminator_; }
    virtual JetRef jet() const { return jet_; }
    virtual TrackRefs selectedTracks() const { return selectedTracks_;}
    virtual TauJetTag* clone() const { return new TauJetTag( *this ); }

    float getDummy() { return dummy_; }
    void setDummy(float d) { dummy_ = d;}
    
  

  private:
    
    float dummy_;
    float discriminator_;
    
    JetRef jet_;
    TrackRefs selectedTracks_;
		    
    
  };
}

#endif
