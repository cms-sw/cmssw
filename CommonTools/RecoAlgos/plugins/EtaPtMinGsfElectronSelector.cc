/* \class EtaPtMinGsfElectronSelector
 *
 * selects electron above a minumum pt cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/AndSelector.h"
#include "CommonTools/UtilAlgos/interface/PtMinSelector.h"
#include "CommonTools/UtilAlgos/interface/EtaRangeSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

 typedef SingleObjectSelector<
           reco::GsfElectronCollection,
           AndSelector<
             EtaRangeSelector,
             PtMinSelector
           >
         > EtaPtMinGsfElectronSelector;

DEFINE_FWK_MODULE( EtaPtMinGsfElectronSelector );
