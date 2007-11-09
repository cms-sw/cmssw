/* 
 *
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "PhysicsTools/RecoAlgos/plugins/EtMinSuperClusterSelector.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

typedef ObjectCountFilter<
          reco::SuperClusterCollection, 
          reco::modules::EtMinSuperClusterSelector
        > EtMinSuperClusterCountFilter;

DEFINE_FWK_MODULE( EtMinSuperClusterCountFilter );
