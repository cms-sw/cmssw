/* \class PtMinElectronViewCandSelector
 *
 * selects electron above a minumum pt cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/View.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"

typedef SingleObjectSelector<
          edm::View<reco::Electron>, 
          PtMinSelector,
          reco::CandidateCollection
        > PtMinElectronViewCandSelector;

DEFINE_FWK_MODULE( PtMinElectronViewCandSelector );
