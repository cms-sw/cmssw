/* \class EtMinCaloJetSelector
 *
 * selects calo-jet above a minumum Et cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/EtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::CaloJetCollection, 
             EtMinSelector<reco::CaloJet> 
           > 
         > EtMinCaloJetSelector;

DEFINE_FWK_MODULE( EtMinCaloJetSelector );
