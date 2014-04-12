/* \class PtMinMuonCountFilter
 *
 * Filters events if at least N muons above 
 * a pt cut are present
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "CommonTools/UtilAlgos/interface/PtMinSelector.h"

typedef ObjectCountFilter<
          reco::MuonCollection, 
          PtMinSelector
        >::type PtMinMuonCountFilter;

DEFINE_FWK_MODULE( PtMinMuonCountFilter );
