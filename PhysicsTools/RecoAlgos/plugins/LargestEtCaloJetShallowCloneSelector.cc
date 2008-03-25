/* \class LargestEtCaloJetShallowCloneSelector
 *
 * selects the N calo-jets with largest Et
 * and save a collection of ShallowCloneCandidate objects
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/interface/ObjectShallowCloneSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SortCollectionSelector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "PhysicsTools/Utilities/interface/EtComparator.h"

 typedef ObjectShallowCloneSelector<
           SortCollectionSelector<
             reco::CaloJetCollection, 
             GreaterByEt<reco::CaloJet>,
             edm::RefVector<reco::CaloJetCollection>
           > 
         > LargestEtCaloJetShallowCloneSelector;

DEFINE_FWK_MODULE( LargestEtCaloJetShallowCloneSelector );
