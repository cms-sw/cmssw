#include "PhysicsTools/CandAlgos/interface/CompositeCandSelector.h"
#include "PhysicsTools/UtilAlgos/interface/AndPairSelector.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "FWCore/Framework/interface/MakerMacros.h"

typedef SingleObjectSelector<
          reco::CandidateView,
          CompositeCandSelector<
            AndPairSelector<
              StringCutObjectSelector<reco::Muon>,
              StringCutObjectSelector<reco::Muon> 
            >,
            reco::Muon
          >
        > CompositeDimuonAndSelector;

DEFINE_FWK_MODULE(CompositeDimuonAndSelector);
