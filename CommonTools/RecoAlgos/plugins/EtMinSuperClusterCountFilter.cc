/* 
 *
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "CommonTools/RecoAlgos/plugins/EtMinSuperClusterSelector.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

typedef ObjectCountFilter<
          reco::SuperClusterCollection, 
          reco::modules::EtMinSuperClusterSelector
        >::type EtMinSuperClusterCountFilter;

DEFINE_FWK_MODULE( EtMinSuperClusterCountFilter );
