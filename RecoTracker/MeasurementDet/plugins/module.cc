#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/MeasurementDet/plugins/MeasurementTrackerESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/typelookup.h"


DEFINE_FWK_EVENTSETUP_MODULE(MeasurementTrackerESProducer);
