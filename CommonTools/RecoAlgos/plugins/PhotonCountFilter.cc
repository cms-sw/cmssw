/* \class PhotonCountFilter
 *
 * Filters events if at least N photons
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"

 typedef ObjectCountFilter<
           reco::PhotonCollection
         >::type PhotonCountFilter;

DEFINE_FWK_MODULE( PhotonCountFilter );
