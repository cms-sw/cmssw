/* \class EtMinPFJetSelector
 *
 * selects pf-jet above a minumum Et cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/EtMinSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/JetReco/interface/PFJet.h"

 typedef SingleObjectSelector<
           reco::PFJetCollection, 
           EtMinSelector
         > EtMinPFJetSelector;

DEFINE_FWK_MODULE( EtMinPFJetSelector );
