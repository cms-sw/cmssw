/* \class EtMinGenJetSelector
 *
 * selects gen-jet above a minumum Et cut
 *
 * \author: Attilio Santocchia, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/EtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/JetReco/interface/GenJet.h"

 typedef SingleObjectSelector<
           reco::GenJetCollection, 
           EtMinSelector
         > EtMinGenJetSelector;

DEFINE_FWK_MODULE( EtMinGenJetSelector );
