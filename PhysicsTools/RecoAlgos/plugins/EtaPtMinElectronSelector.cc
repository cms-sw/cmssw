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
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"

 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::ElectronCollection,
             AndSelector<
               EtaRangeSelector<reco::Electron>,
               PtMinSelector<reco::Electron> 
             >
           >
         > EtaPtMinElectronSelector;

DEFINE_FWK_MODULE( EtaPtMinElectronSelector );
