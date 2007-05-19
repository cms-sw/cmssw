/* \class PtMinJetSelector
 * 
 * Jet Selector based on a minimum pt cut.
 * Usage:
 * 
 * module selectedJetss = PtMinXXXJetSelector {
 *   InputTag src = myCollection
 *   double ptMin = 15.0
 * };
 * where "XXX" is "Calo", or "Gen", or "PF", or "Basic"
 *
 * \author: Fedor Ratnikov, UMd. Inherited from PtMinCandSelector by Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"

typedef SingleObjectSelector <reco::CaloJetCollection, PtMinSelector<reco::CaloJet> > PtMinCaloJetSelector;
typedef SingleObjectSelector <reco::GenJetCollection, PtMinSelector<reco::GenJet> > PtMinGenJetSelector;
typedef SingleObjectSelector <reco::PFJetCollection, PtMinSelector<reco::PFJet> > PtMinPFJetSelector;
typedef SingleObjectSelector <reco::BasicJetCollection, PtMinSelector<reco::BasicJet> > PtMinBasicJetSelector;

