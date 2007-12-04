/* \class EtaPtMinPixelMatchGsfElectronSelector
 *
 * selects electron above a minumum pt cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/AndSelector.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/EtaRangeSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"

 typedef SingleObjectSelector<
           reco::PixelMatchGsfElectronCollection,
           AndSelector<
             EtaRangeSelector,
             PtMinSelector
           >
         > EtaPtMinPixelMatchGsfElectronSelector;

DEFINE_FWK_MODULE( EtaPtMinPixelMatchGsfElectronSelector );
