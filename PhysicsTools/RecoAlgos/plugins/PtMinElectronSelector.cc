/* \class PtMinElectronSelector
 *
 * selects electron above a minumum pt cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"

 typedef SingleObjectSelector<
           reco::ElectronCollection, 
           PtMinSelector<reco::Electron> 
         > PtMinElectronSelector;

DEFINE_FWK_MODULE( PtMinElectronSelector );
