//#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

using namespace cms;

#include "RecoHI/HiMuonAlgos/plugins/HLTHIMuL1L2L3Filter.h"
#include "RecoHI/HiMuonAlgos/plugins/TestMuL1L2Filter.h"
using cms::HLTHIMuL1L2L3Filter;
using cms::TestMuL1L2Filter;
DEFINE_FWK_MODULE(HLTHIMuL1L2L3Filter);
DEFINE_FWK_MODULE(TestMuL1L2Filter);

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "HIMuonTrackingRegionProducer.h"
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, HIMuonTrackingRegionProducer, "HIMuonTrackingRegionProducer");
