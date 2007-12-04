/* \class EtaPtMinPixelMatchGsfElectronFullCloneSelector
 *
 * selects electron above a minumum pt cut
 * Also clones super cluster and track of selected electrons
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
#include "PhysicsTools/RecoAlgos/interface/PixelMatchGsfElectronSelector.h"

 typedef SingleObjectSelector<
           reco::PixelMatchGsfElectronCollection,
           AndSelector<
	     EtaRangeSelector,
             PtMinSelector
           >
         > EtaPtMinPixelMatchGsfElectronFullCloneSelector;

DEFINE_FWK_MODULE( EtaPtMinPixelMatchGsfElectronFullCloneSelector );
