#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/btau/interface/HLTJetTag.h"
#include "HLTrigger/btau/interface/HLTTauL25DoubleFilter.h"
#include "HLTrigger/btau/interface/HLTDisplacedmumuFilter.h"
#include "HLTrigger/btau/interface/HLTL1MuonCorrector.h"
#include "HLTrigger/btau/interface/HLTmumuGammaFilter.h"
#include "HLTrigger/btau/interface/GetJetsFromHLTobject.h"

DEFINE_FWK_MODULE(HLTJetTag);
DEFINE_FWK_MODULE(HLTTauL25DoubleFilter);
DEFINE_FWK_MODULE(HLTDisplacedmumuFilter);
DEFINE_FWK_MODULE(HLTL1MuonCorrector);
DEFINE_FWK_MODULE(HLTmumuGammaFilter);
DEFINE_FWK_MODULE(GetJetsFromHLTobject);
