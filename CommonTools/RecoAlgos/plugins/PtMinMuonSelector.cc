/* \class PtMinMuonSelector
 *
 * selects muon above a minumum pt cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/PtMinSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

 typedef SingleObjectSelector<
           reco::MuonCollection, 
           PtMinSelector
         > PtMinMuonSelector;

DEFINE_FWK_MODULE( PtMinMuonSelector );
