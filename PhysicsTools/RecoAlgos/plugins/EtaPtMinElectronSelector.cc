/* \class EtaPtMinElectronSelector
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
#include "DataFormats/EgammaCandidates/interface/Electron.h"

 typedef SingleObjectSelector<
           reco::ElectronCollection,
           AndSelector<
             EtaRangeSelector<reco::Electron>,
             PtMinSelector<reco::Electron> 
           >
         > EtaPtMinElectronSelector;

DEFINE_FWK_MODULE( EtaPtMinElectronSelector );
