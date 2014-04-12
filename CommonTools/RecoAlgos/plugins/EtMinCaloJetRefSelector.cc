/* \class EtMinCaloJetRefSelector
 *
 * selects calo-jet above a minumum Et cut 
 * and saves a collection of references (RefVector) to selctedobjects
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
           EtMinSelector,
           reco::CaloJetRefVector
         > EtMinCaloJetRefSelector;

DEFINE_FWK_MODULE( EtMinCaloJetRefSelector );
