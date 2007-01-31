/* \class PtMinElectronSelector
 *
 * selects electron above a minumum pt cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"

 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::ElectronCollection, 
             PtMinSelector<reco::Electron> 
           > 
         > PtMinElectronSelector;

DEFINE_FWK_MODULE( PtMinElectronSelector );
