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
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"

 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::PixelMatchGsfElectronCollection,
	     AndSelector<
	       EtaRangeSelector<reco::PixelMatchGsfElectron>,
               PtMinSelector<reco::PixelMatchGsfElectron> 
             >
           >
         > EtaPtMinPixelMatchGsfElectronSelector;

DEFINE_FWK_MODULE( EtaPtMinPixelMatchGsfElectronSelector );
