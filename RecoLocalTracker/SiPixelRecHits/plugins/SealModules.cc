//
// plugins/SealModules.cc
// Using new EDM plugin manager (V.Chiochia, April 2007)
//
//--- Our components which we want Framework to know about:
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
//--- The CPE ES Producers
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGenericESProducer.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPETemplateRecoESProducer.h"
//---- The RecHit ED producer
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelRecHitConverter.h"
//--- The header files for the Framework infrastructure (macros etc):
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

using cms::SiPixelRecHitConverter;

DEFINE_FWK_MODULE(SiPixelRecHitConverter);
DEFINE_FWK_EVENTSETUP_MODULE(PixelCPEGenericESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(PixelCPETemplateRecoESProducer);

