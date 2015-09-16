#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// To be used in the future for any cut
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "HLTrigger/Egamma/interface/HLTEgammaGenericQuadraticFilter.h"
#include "HLTrigger/Egamma/interface/HLTEgammaGenericQuadraticEtaFilter.h"
#include "HLTrigger/Egamma/interface/HLTElectronGenericFilter.h"

#include "HLTrigger/Egamma/interface/HLTEgammaEtFilter.h"
#include "HLTrigger/Egamma/interface/HLTEgammaDoubleEtFilter.h"
#include "HLTrigger/Egamma/interface/HLTElectronPixelMatchFilter.h"
#include "HLTrigger/Egamma/interface/HLTPMMassFilter.h"
#include "HLTrigger/Egamma/interface/HLTElectronMuonInvMassFilter.h"
#include "HLTrigger/Egamma/interface/HLTPMDocaFilter.h"

#include "HLTrigger/Egamma/interface/HLTEgammaL1MatchFilterRegional.h"
#include "HLTrigger/Egamma/interface/HLTElectronEoverpFilterRegional.h"

#include "HLTrigger/Egamma/interface/HLTEgammaDoubleEtPhiFilter.h"
#include "HLTrigger/Egamma/interface/HLTElectronOneOEMinusOneOPFilterRegional.h"

#include "HLTrigger/Egamma/interface/HLTEgammaL1MatchFilterPairs.h"
#include "HLTrigger/Egamma/interface/HLTEgammaEtFilterPairs.h"
#include "HLTrigger/Egamma/interface/HLTEgammaCaloIsolFilterPairs.h"
#include "HLTrigger/Egamma/interface/HLTEgammaTriggerFilterObjectWrapper.h"
#include "HLTrigger/Egamma/interface/HLTElectronEtFilter.h"
#include "HLTrigger/Egamma/interface/HLTEgammaDoubleEtDeltaPhiFilter.h"
#include "HLTrigger/Egamma/interface/HLTEgammaDoubleLegCombFilter.h"
#include "HLTrigger/Egamma/interface/HLTEgammaCombMassFilter.h"
#include "HLTrigger/Egamma/interface/HLTEgammaAllCombMassFilter.h"

#include "HLTrigger/Egamma/interface/HLTDisplacedEgammaFilter.h"
#include "HLTrigger/Egamma/interface/HLTElectronMissingHitsFilter.h"

#include "HLTrigger/Egamma/interface/HLTElectronPFMTFilter.h"
#include "HLTrigger/Egamma/src/HLTElectronPFMTFilter.cc"

#include "HLTrigger/Egamma/interface/HLTScoutingEgammaProducer.h"

DEFINE_FWK_MODULE(HLTEgammaGenericQuadraticFilter);
DEFINE_FWK_MODULE(HLTEgammaGenericQuadraticEtaFilter);
DEFINE_FWK_MODULE(HLTEgammaEtFilter);
DEFINE_FWK_MODULE(HLTEgammaDoubleEtFilter);
DEFINE_FWK_MODULE(HLTElectronPixelMatchFilter);

typedef HLTElectronPFMTFilter<reco::Electron> HLTGsfElectronPFMTFilter;
DEFINE_FWK_MODULE(HLTGsfElectronPFMTFilter);
typedef HLTElectronPFMTFilter<reco::RecoEcalCandidate> HLTEcalCandidatePFMTFilter;
DEFINE_FWK_MODULE(HLTEcalCandidatePFMTFilter);


DEFINE_FWK_MODULE(HLTPMMassFilter);
DEFINE_FWK_MODULE(HLTElectronMuonInvMassFilter);
DEFINE_FWK_MODULE(HLTPMDocaFilter);

DEFINE_FWK_MODULE(HLTEgammaL1MatchFilterRegional);
DEFINE_FWK_MODULE(HLTElectronEoverpFilterRegional);

DEFINE_FWK_MODULE(HLTEgammaDoubleEtPhiFilter);

DEFINE_FWK_MODULE(HLTElectronGenericFilter);
DEFINE_FWK_MODULE(HLTElectronOneOEMinusOneOPFilterRegional);
DEFINE_FWK_MODULE(HLTEgammaL1MatchFilterPairs);
DEFINE_FWK_MODULE(HLTEgammaEtFilterPairs);
DEFINE_FWK_MODULE(HLTEgammaCaloIsolFilterPairs);
DEFINE_FWK_MODULE(HLTEgammaTriggerFilterObjectWrapper);
DEFINE_FWK_MODULE(HLTElectronEtFilter);
DEFINE_FWK_MODULE(HLTEgammaDoubleEtDeltaPhiFilter);
DEFINE_FWK_MODULE(HLTEgammaDoubleLegCombFilter);
DEFINE_FWK_MODULE(HLTEgammaCombMassFilter);
DEFINE_FWK_MODULE(HLTEgammaAllCombMassFilter);
DEFINE_FWK_MODULE(HLTDisplacedEgammaFilter);

DEFINE_FWK_MODULE(HLTElectronMissingHitsFilter);

DEFINE_FWK_MODULE(HLTScoutingEgammaProducer);
