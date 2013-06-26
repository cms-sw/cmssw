/* \class MuonCountFilter
 *
 * Filters events if at least N muons
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"

 typedef ObjectCountFilter<
           reco::MuonCollection
         >::type MuonCountFilter;

DEFINE_FWK_MODULE( MuonCountFilter );
