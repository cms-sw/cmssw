#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "CalibTracker/SiPixelConnectivity/interface/PixelToFEDAssociate.h"
#include "PixelToFEDAssociateFromAsciiESProducer.h"
#include "PixelToLNKAssociateFromAsciiESProducer.h"

EVENTSETUP_DATA_REG(PixelToFEDAssociate);
DEFINE_FWK_EVENTSETUP_MODULE(PixelToFEDAssociateFromAsciiESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(PixelToLNKAssociateFromAsciiESProducer);
