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
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/PtMinSelector.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"

typedef SingleObjectSelector <reco::CaloJetCollection, PtMinSelector> PtMinCaloJetSelector;
typedef SingleObjectSelector <reco::GenJetCollection, PtMinSelector> PtMinGenJetSelector;
typedef SingleObjectSelector <reco::PFJetCollection, PtMinSelector> PtMinPFJetSelector;
typedef SingleObjectSelector <reco::BasicJetCollection, PtMinSelector> PtMinBasicJetSelector;

