/* \class EtMinPhotonSelector
 *
 * selects photon above a minumum Et cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/EtMinSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

 typedef SingleObjectSelector<
           reco::PhotonCollection, 
           EtMinSelector
         > EtMinPhotonSelector;

DEFINE_FWK_MODULE( EtMinPhotonSelector );
