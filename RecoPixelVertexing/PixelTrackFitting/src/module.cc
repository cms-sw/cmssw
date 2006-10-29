#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "PixelFitterByConformalMappingAndLineESProducer.h"
#include "PixelFitterByHelixProjectionsESProducer.h"

EVENTSETUP_DATA_REG(PixelFitter);
DEFINE_FWK_EVENTSETUP_MODULE(PixelFitterByConformalMappingAndLineESProducer)
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(PixelFitterByHelixProjectionsESProducer)

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackProducer.h"

DEFINE_ANOTHER_FWK_MODULE(PixelTrackProducer)
