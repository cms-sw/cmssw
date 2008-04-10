#ifndef __PFBlockElement__
#define __PFBlockElement__

#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFNuclearInteraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"


#include <iostream>


namespace reco {
  class PFBlockElementCluster;
  class PFBlockElementTrack;
  
  
  /// \brief Abstract base class for a PFBlock element (track, cluster...)
  /// 
  /// this class contains a PFRecTrackRef of a 
  /// PFClusterRef, depending on the type of the element
  class PFBlockElement {
  public:
    
    /// number of element types
    static const unsigned nTypes_;

    /// possible types for the element
    enum Type { 
      NONE=0,
      TRACK, 
      PS1, 
      PS2, 
      ECAL, 
      HCAL 
    };

    enum TrackType {
      DEFAULT=0,
      T_FROM_GSF,  
      T_FROM_BREM, 
      T_FROM_NUCL,
      T_TO_NUCL,
      T_FROM_GAMMACONV,
      MUON
    };

    /// standard constructor
    PFBlockElement(Type type=NONE) :  
      type_(type), 
      locked_(false),
      index_( static_cast<unsigned>(-1) ) {
    }


    /// destructor
    virtual ~PFBlockElement() {}
  
    /// print the object inside the element
    virtual void Dump(std::ostream& out=std::cout, 
                      const char* tab=" " ) const;
    
    /// necessary to have the edm::OwnVector<PFBlockElement> working
    virtual PFBlockElement* clone() const = 0;
      
    /// lock element
    void lock() {locked_ = true;}

    /// unlock element
    void unLock() {locked_ = false;}

    /// \return type
    Type type() const { return type_; }

    /// \return tracktype
    virtual TrackType trackType() const { return DEFAULT; }

    /// locked ? 
    bool    locked() const {return locked_;}
    
    /// set index 
    void     setIndex(unsigned index) { index_ = index; }

    /// \return index
    unsigned index() const {return index_;} 

    virtual reco::TrackRef trackRef()  const {return reco::TrackRef(); }
    virtual PFRecTrackRef trackRefPF()  const {return PFRecTrackRef(); }
    virtual PFClusterRef clusterRef() const {return PFClusterRef(); }
    virtual PFNuclearInteractionRef nuclearRef() const { return PFNuclearInteractionRef(); }

    virtual bool isSecondary() const { return false; }

    friend std::ostream& operator<<( std::ostream& out, 
                                     const PFBlockElement& element );
    
  protected:  
  
    /// type, see PFBlockElementType
    Type     type_;

    /// locked flag. 
    /// \todo can probably be transient. Could be replaced by a 
    /// "remaining energy"
    bool       locked_;
    
    /// index in block vector 
    unsigned   index_;

  };
}
#endif
