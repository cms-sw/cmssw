/* \class EtMinCaloJetSelector
 *
 * selects calo-jet above a minumum Et cut 
 * and saves a collection of ShallowCloneCandidate objects
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/EtMinSelector.h"
#include "PhysicsTools/CandAlgos/interface/ObjectShallowCloneSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

 typedef ObjectShallowCloneSelector<
           SingleElementCollectionSelector<
             reco::CaloJetCollection,
             EtMinSelector<reco::CaloJet>,
             edm::RefVector<reco::CaloJetCollection>
           >
         > EtMinCaloJetShallowCloneSelector;

DEFINE_FWK_MODULE( EtMinCaloJetShallowCloneSelector );
