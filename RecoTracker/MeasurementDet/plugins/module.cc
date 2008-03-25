#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/MeasurementDet/plugins/MeasurementTrackerESProducer.h"
#include "RecoTracker/MeasurementDet/plugins/MeasurementTrackerSiStripRefGetterProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(MeasurementTrackerESProducer);
DEFINE_ANOTHER_FWK_MODULE(MeasurementTrackerSiStripRefGetterProducer);


#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTracker/MeasurementDet/interface/UpdaterService.h"
DEFINE_FWK_SERVICE( UpdaterService );


