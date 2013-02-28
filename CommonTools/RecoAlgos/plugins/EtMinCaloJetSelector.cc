/* \class EtMinCaloJetSelector
 *
 * selects calo-jet above a minumum Et cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/EtMinSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

 typedef SingleObjectSelector<
           reco::CaloJetCollection, 
           EtMinSelector
         > EtMinCaloJetSelector;

DEFINE_FWK_MODULE( EtMinCaloJetSelector );
