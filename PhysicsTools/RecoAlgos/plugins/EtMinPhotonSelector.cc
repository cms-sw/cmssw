/* \class EtMinPhotonSelector
 *
 * selects photon above a minumum Et cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/EtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

 typedef SingleObjectSelector<
           reco::PhotonCollection, 
           EtMinSelector<reco::Photon> 
         > EtMinPhotonSelector;

DEFINE_FWK_MODULE( EtMinPhotonSelector );
