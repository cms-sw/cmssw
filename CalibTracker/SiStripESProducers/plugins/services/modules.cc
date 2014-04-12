#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripPedestalsGenerator.h"
DEFINE_FWK_SERVICE(SiStripPedestalsGenerator);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripNoisesGenerator.h"
DEFINE_FWK_SERVICE(SiStripNoisesGenerator);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripApvGainGenerator.h"
DEFINE_FWK_SERVICE(SiStripApvGainGenerator);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripLorentzAngleGenerator.h"
DEFINE_FWK_SERVICE(SiStripLorentzAngleGenerator);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripBackPlaneCorrectionGenerator.h"
DEFINE_FWK_SERVICE(SiStripBackPlaneCorrectionGenerator);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripThresholdGenerator.h"
DEFINE_FWK_SERVICE(SiStripThresholdGenerator);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripBadModuleGenerator.h"
DEFINE_FWK_SERVICE(SiStripBadModuleGenerator);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripLatencyGenerator.h"
DEFINE_FWK_SERVICE(SiStripLatencyGenerator);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripBaseDelayGenerator.h"
DEFINE_FWK_SERVICE(SiStripBaseDelayGenerator);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripConfObjectGenerator.h"
DEFINE_FWK_SERVICE(SiStripConfObjectGenerator);
