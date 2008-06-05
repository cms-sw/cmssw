#include "PhysicsTools/PatAlgos/plugins/PATGenericParticleCleaner.h"
#include "PhysicsTools/PatAlgos/plugins/PATGenericParticleCleaner.icc"

#include <DataFormats/RecoCandidate/interface/RecoChargedCandidate.h>
#include <DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h>
#include <DataFormats/RecoCandidate/interface/RecoEcalCandidate.h>

namespace pat {
    typedef PATGenericParticleCleaner<reco::RecoChargedCandidate>   PATRecoChargedCandidateCleaner;
    typedef PATGenericParticleCleaner<reco::RecoEcalCandidate>      PATRecoEcalCandidateCleaner;
    typedef PATGenericParticleCleaner<reco::RecoCaloTowerCandidate> PATRecoCaloTowerCandidateCleaner;
}

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace pat;
DEFINE_FWK_MODULE(PATRecoChargedCandidateCleaner);
DEFINE_FWK_MODULE(PATRecoEcalCandidateCleaner);
DEFINE_FWK_MODULE(PATRecoCaloTowerCandidateCleaner);
