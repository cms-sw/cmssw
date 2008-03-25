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
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "PhysicsTools/UtilAlgos/interface/EtMinSelector.h"

typedef ObjectCountFilter<
          reco::PhotonCollection, 
          EtMinSelector
        > EtMinPhotonCountFilter;

DEFINE_FWK_MODULE( EtMinPhotonCountFilter );
