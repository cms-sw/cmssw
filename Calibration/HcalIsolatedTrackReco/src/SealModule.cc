#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Calibration/HcalIsolatedTrackReco/interface/IsolatedPixelTrackCandidateProducer.h"
#include "Calibration/HcalIsolatedTrackReco/interface/EcalIsolatedParticleCandidateProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h" 	 
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h" 	 
#include "Calibration/HcalIsolatedTrackReco/interface/SiStripRegFEDSelector.h"
#include "Calibration/HcalIsolatedTrackReco/interface/ECALRegFEDSelector.h"
#include "Calibration/HcalIsolatedTrackReco/interface/SubdetFEDSelector.h"
#include "HITRegionalPixelSeedGenerator.h"
#include "Calibration/HcalIsolatedTrackReco/interface/IPTCorrector.h"
#include "Calibration/HcalIsolatedTrackReco/interface/HITSiStripRawToClustersRoI.h" 	 

DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, HITRegionalPixelSeedGenerator, "HITRegionalPixelSeedGenerator"); 
//
DEFINE_FWK_MODULE(IsolatedPixelTrackCandidateProducer);
DEFINE_FWK_MODULE(EcalIsolatedParticleCandidateProducer);
DEFINE_FWK_MODULE(SiStripRegFEDSelector);
DEFINE_FWK_MODULE(ECALRegFEDSelector);
DEFINE_FWK_MODULE(SubdetFEDSelector);
DEFINE_FWK_MODULE(IPTCorrector);
DEFINE_FWK_MODULE(HITSiStripRawToClustersRoI);
