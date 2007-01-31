/* \class EtMinPhotonCountFilter
 *
 * Filters events if at least N photons above 
 * an Et cut are present
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "PhysicsTools/UtilAlgos/interface/EtMinSelector.h"

typedef ObjectCountFilter<
          reco::PhotonCollection, 
          EtMinSelector<reco::Photon>
        > EtMinPhotonCountFilter;

DEFINE_FWK_MODULE( EtMinPhotonCountFilter );
