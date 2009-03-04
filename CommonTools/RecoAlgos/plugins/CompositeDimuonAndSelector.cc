#include "CommonTools/CandAlgos/interface/CompositeCandSelector.h"
#include "CommonTools/UtilAlgos/interface/AndPairSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
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
