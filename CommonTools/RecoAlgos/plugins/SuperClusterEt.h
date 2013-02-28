#ifndef RecoAlgos_Superclustere_h
#define RecoAlgos_Superclustere_h

#include "DataFormats/EgammaReco/interface/SuperCluster.h"

   struct SuperClusterEt {
     typedef reco::SuperCluster type;
     double operator()( const reco::SuperCluster & c ) const {
       return c.energy() * sin( c.position().theta() );
     }
   };



#endif
