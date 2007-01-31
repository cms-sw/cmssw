/* \class EtMinPhotonSelector
 *
 * selects photon above a minumum Et cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/EtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::PhotonCollection, 
             EtMinSelector<reco::Photon> 
           > 
         > EtMinPhotonSelector;

DEFINE_FWK_MODULE( EtMinPhotonSelector );
