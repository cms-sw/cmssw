#ifndef __PFBlockElement__
#define __PFBlockElement__

#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"


#include <iostream>


namespace reco {
  class PFBlockElementCluster;
  class PFBlockElementTrack;
  
  
  /// \brief Base element of a PFBlock (track, cluster...)
  /// 
  /// this class is essentially wraps a PFRecTrackRef of a 
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
      HCAL, 
      MUON
    };
    

    /// default constructor 
    PFBlockElement() 
      :  
      type_( NONE ), 
      locked_(false), 
      index_( static_cast<unsigned>(-1) ) {
/*       ++instanceCounter_; */
    }

    /// standard constructor
    PFBlockElement(Type type) 
      :  
      type_(type), 
      locked_(false),
      index_( static_cast<unsigned>(-1) ) {
/*       ++instanceCounter_;     */
    }

    /// copy constructor
    /// \todo remove when instanceCounter is not needed anymore
/*     PFBlockElement(const PFBlockElement& other)  */
/*       :   */
/*       type_( other.type_ ),  */
/*       locked_( other.locked_ ), */
/*       index_( other.index_ )  { */
/*       ++instanceCounter_; */
/*     } */

    /// destructor
    virtual ~PFBlockElement() {
/*       --instanceCounter_;         */
    }
  
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

    /// locked ? 
    bool    locked() const {return locked_;}
    
    /// set index 
    void     setIndex(unsigned index) { index_ = index; }

    /// \return index
    unsigned index() const {return index_;} 

/*     static int instanceCounter(); */
 
    /// supplement dynamic_cast, but faster
    virtual const PFBlockElementCluster*   toCluster() const 
      {return (const PFBlockElementCluster*)0; }
  
    /// supplement dynamic_cast, but faster
    virtual const PFBlockElementTrack*     toTrack() const 
      {return (const PFBlockElementTrack*)0; }

    virtual PFRecTrackRef trackRef()  const {return PFRecTrackRef();}
    virtual PFClusterRef  clusterRef() const {return PFClusterRef();}


    friend std::ostream& operator<<( std::ostream& out, 
				     const PFBlockElement& element );
    
  protected:
    
    /// block, not owner
    // const PFBlock*              pfBlock_;   
  
    /// links,
    /// first is a pointer to the element at the other side. 
    /// second is the slot in the link vector of pfBlock_
    // std::map< const PFBlockElement*, unsigned >      links_;   
  
    /// type
    Type     type_;
  
    /// locked ? can probably be transient. should be replaced by a "remaining energy"
    bool       locked_;
    
    /// index in block vector 
    unsigned   index_;

/*     static int        instanceCounter_; */
  };
}
#endif
