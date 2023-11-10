#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"

namespace l1ScoutingRun3 { 
  
  edm::Wrapper<ScMuonOrbitCollection>    ScMuonOrbitCollectionWrapper;
  edm::Wrapper<ScJetOrbitCollection>     ScJetOrbitCollectionWrapper;
  edm::Wrapper<ScEGammaOrbitCollection>  ScEGammaOrbitCollectionWrapper;
  edm::Wrapper<ScTauOrbitCollection>     ScTauOrbitCollectionWrapper;
  edm::Wrapper<ScEtSumOrbitCollection>   ScEtSumOrbitCollectionWrapper;
  edm::Wrapper<ScBxSumsOrbitCollection>  ScBxSumsOrbitCollectionWrapper;
}