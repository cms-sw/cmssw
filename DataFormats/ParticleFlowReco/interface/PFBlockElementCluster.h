#ifndef __PFBlockElementCluster__
#define __PFBlockElementCluster__

#include <iostream>

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

namespace reco {
  
  /// \brief Cluster Element.
  /// 
  /// this class contains a reference to a PFCluster 
  class PFBlockElementCluster : public PFBlockElement {
  public:
    PFBlockElementCluster() {} 
    
    /// \brief constructor.
    /// type must be equal to PS1, PS2, ECAL, HCAL. 
    /// \todo add a protection against the other types...
    PFBlockElementCluster(const PFClusterRef& ref, 
                          PFBlockElement::Type type) 
      : 
      PFBlockElement(type),
      clusterRef_( ref ) {}
    
    PFBlockElement* clone() const { return new PFBlockElementCluster(*this); }
    
    /// \return reference to the corresponding cluster
    PFClusterRef  clusterRef() const {return clusterRef_;}

    void Dump(std::ostream& out = std::cout, 
              const char* tab = " " ) const;
  
  private:
    /// reference to the corresponding cluster
    PFClusterRef  clusterRef_;
  };
}

#endif

