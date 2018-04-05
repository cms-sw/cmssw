#ifndef DataFormats_ParticleFlowReco_PFLayer_h
#define DataFormats_ParticleFlowReco_PFLayer_h

#include "DataFormats/CaloRecHit/interface/CaloID.h"


/**\class PFLayer
   \brief layer definition for PFRecHit and PFCluster
   
   These definitions are intended for internal use in the particle 
   flow packages.
   
   \todo A new layer definition has been provided in reco::CaloID.
   translation functions have been added between reco::CaloID and 
   PFLayer, but PFLayer should eventually be phased out completely
   
   \author Colin Bernet
   \date   July 2006
*/

class PFLayer {

 public:
  /// constructor
  PFLayer() {}

  /// destructor
  ~PFLayer() {}

  /// layer definition
  enum Layer {PS2          = -12, 
              PS1          = -11,
              ECAL_ENDCAP  = -2,
              ECAL_BARREL  = -1,
	      NONE         = 0,
              HCAL_BARREL1 = 1,
              HCAL_BARREL2 = 2,
              HCAL_ENDCAP  = 3,
              HF_EM        = 11,
              HF_HAD       = 12,
              HGCAL        = 13  // HGCal, could be EM or HAD
  };

  static reco::CaloID  toCaloID( Layer layer);
  
  static Layer         fromCaloID( const reco::CaloID& id);
};

#endif
