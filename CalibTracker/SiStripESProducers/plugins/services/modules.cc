#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_SEAL_MODULE();


#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripPedestalsGenerator.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripPedestalsGenerator);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripNoisesGenerator.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripNoisesGenerator);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripApvGainGenerator.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripApvGainGenerator);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripLorentzAngleGenerator.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripLorentzAngleGenerator);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripThresholdGenerator.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripThresholdGenerator);

