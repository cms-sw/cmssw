/* \class PtMinElectronCountFilter
 *
 * Filters events if at least N electrons above 
 * a pt cut are present
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "CommonTools/UtilAlgos/interface/PtMinSelector.h"

typedef ObjectCountFilter<
          reco::ElectronCollection, 
          PtMinSelector
        >::type PtMinElectronCountFilter;

DEFINE_FWK_MODULE( PtMinElectronCountFilter );
