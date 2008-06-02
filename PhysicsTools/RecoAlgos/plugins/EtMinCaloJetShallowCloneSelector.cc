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
#include "PhysicsTools/CandAlgos/interface/SingleObjectShallowCloneSelector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

 typedef SingleObjectShallowCloneSelector<
           reco::CaloJetCollection,
           EtMinSelector
         > EtMinCaloJetShallowCloneSelector;

DEFINE_FWK_MODULE( EtMinCaloJetShallowCloneSelector );
