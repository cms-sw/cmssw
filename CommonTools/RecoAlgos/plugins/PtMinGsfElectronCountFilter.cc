/* \class PtMinGsfElectronCountFilter
 *
 * Filters events if at least N electrons above 
 * a pt cut are present
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "CommonTools/UtilAlgos/interface/PtMinSelector.h"

typedef ObjectCountFilter<
          reco::GsfElectronCollection, 
          PtMinSelector
        >::type PtMinGsfElectronCountFilter;

DEFINE_FWK_MODULE( PtMinGsfElectronCountFilter );
