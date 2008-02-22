/* \class EtaPtMinGsfElectronFullCloneSelector
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
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "PhysicsTools/RecoAlgos/interface/GsfElectronSelector.h"

 typedef SingleObjectSelector<
           reco::GsfElectronCollection,
           AndSelector<
	     EtaRangeSelector,
             PtMinSelector
           >
         > EtaPtMinGsfElectronFullCloneSelector;

DEFINE_FWK_MODULE( EtaPtMinGsfElectronFullCloneSelector );
