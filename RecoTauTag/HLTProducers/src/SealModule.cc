//#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTauTag/HLTProducers/interface/IsolatedTauJetsSelector.h"
#include "RecoTauTag/HLTProducers/interface/EMIsolatedTauJetsSelector.h"
#include "RecoTauTag/HLTProducers/interface/L2TauJetsProvider.h"
#include "RecoTauTag/HLTProducers/interface/L2TauJetsMerger.h"
#include "RecoTauTag/HLTProducers/interface/HLTTauProducer.h"
#include "RecoTauTag/HLTProducers/interface/CaloTowerCreatorForTauHLT.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h" 	 
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h" 	 
#include "TauRegionalPixelSeedGenerator.h" 	 
#include "RecoTauTag/HLTProducers/interface/L2TauIsolationSelector.h"
#include "RecoTauTag/HLTProducers/interface/L2TauIsolationProducer.h"



 	 
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, TauRegionalPixelSeedGenerator, "TauRegionalPixelSeedGenerator"); 	 

//DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(IsolatedTauJetsSelector);
DEFINE_ANOTHER_FWK_MODULE(EMIsolatedTauJetsSelector);
DEFINE_ANOTHER_FWK_MODULE(L2TauJetsProvider);
DEFINE_ANOTHER_FWK_MODULE(L2TauJetsMerger);
DEFINE_ANOTHER_FWK_MODULE(CaloTowerCreatorForTauHLT);
DEFINE_ANOTHER_FWK_MODULE(HLTTauProducer);
DEFINE_ANOTHER_FWK_MODULE(L2TauIsolationProducer);
DEFINE_ANOTHER_FWK_MODULE(L2TauIsolationSelector);
