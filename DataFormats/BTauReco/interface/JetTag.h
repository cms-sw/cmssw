#ifndef BTauReco_JetTag_h
#define BTauReco_JetTag_h
// \class JetTag
// 
// \short base class for persistent tagging result 
// JetTag is an pure virtual interface class. Base class for result of all b- and tau-tagging algorithms.
// 
//
// \author Marcel Vos, based on ORCA version by Christian Weiser, Andrea Rizzi
// \version first version on January 12, 2006

#include "DataFormats/JetObjects/interface/Jet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <vector>
#include "FWCore/EDProduct/interface/Ref.h"
#include "FWCore/EDProduct/interface/RefVector.h"



typedef edm::Ref<std::vector<Jet> > JetRef ; /// MVS: Jet does not have name sp\ace reco and does no forward typedefs



namespace reco {
  class JetTag  {

//
// The base class for jets with tag information.
// Concrete b- and tau tag Algos extend this interface with additional, algo-specific info
//


 public:

  
    JetTag() {}

    
    virtual ~JetTag() {} 
    
    // discriminator should be a continuous variable, the tagging efficiency and purity should have a monotonous dependence on the discriminator

    virtual float discriminator () const=0;  
  
    // reference to jet

    virtual JetRef jetRef () const=0;  // 
    virtual const Jet & jet() const=0; 

    // reference to all tracks that have passed selection and thus contribute to the discriminator  
    virtual TrackRefs selectedTracks () const=0; 

    virtual JetTag* clone() const=0;

    
 private:
    

   
};
}
#endif
