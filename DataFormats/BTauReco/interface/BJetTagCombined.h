#ifndef BTauReco_BJetTagCombined_h
#define BTauReco_BJetTagCombined_h
//
// \class BJetTagCombined
// \short concrete class for the description of the result of the combined b-tagging algorithm 
//
// concrete class inherits from JetTag
// contains the result of combined b-tagging algorithm 
// object to be made persistent on RECO
//
// \author: Christian Weiser, Andrea Rizzi, Marcel Vos
//

#include "DataFormats/BTauReco/interface/JetTag.h"

namespace reco {
  class BJetTagCombined : public JetTag {
 
  //
  // information specific for combined SV algo
  // (just one example; typically each algorithm extends the base object)
  //
    
  public:


    explicit BJetTagCombined() {}
    explicit BJetTagCombined( const JetTag & p ) : JetTag(p) {}    
    BJetTagCombined(float discriminator, JetRef jet, TrackRefs tracks) : discriminator_ (discriminator), jet_(jet), selectedTracks_ (tracks)  {}


    virtual float discriminator() const { return discriminator_; }
    virtual JetRef jetRef() const { return jet_; }
    virtual const Jet & jet() const { return *jet_; }
    virtual TrackRefs selectedTracks() const { return selectedTracks_;}
    virtual BJetTagCombined* clone() const { return new BJetTagCombined( *this ); }
    // get additional info 
    //  BTagMultiVariate multiVariate () const { return multiVariate_ ; }
    float getDummy() { return dummy_; }
    void setDummy(float d) { dummy_ = d;}
    
  

  private:
    
    // the additional variables are stored in a simple 'container class';
    // it contains all the inputs needed to compute the combined discriminator
    // (track impact parameter significances; vertex mass, multiplicity etc.)
    // BTagMultiVariate multiVariate_ ;
    
    float dummy_;
    float discriminator_;
    
    JetRef jet_;
    TrackRefs selectedTracks_;
		    
    
  };
}

#endif
