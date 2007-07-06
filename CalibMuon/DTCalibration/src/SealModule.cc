#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "CalibMuon/DTCalibration/src/DTMapGenerator.h"
#include "CalibMuon/DTCalibration/src/DTTTrigCalibration.h"
#include "CalibMuon/DTCalibration/src/DTTTrigWriter.h"
#include "CalibMuon/DTCalibration/src/DTT0Calibration.h"
#include "CalibMuon/DTCalibration/src/DTVDriftCalibration.h"
#include "CalibMuon/DTCalibration/src/DTVDriftWriter.h"
#include "CalibMuon/DTCalibration/src/DTFakeTTrigESProducer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(DTMapGenerator);
DEFINE_ANOTHER_FWK_MODULE(DTTTrigCalibration);
DEFINE_ANOTHER_FWK_MODULE(DTTTrigWriter);
DEFINE_ANOTHER_FWK_MODULE(DTT0Calibration);
DEFINE_ANOTHER_FWK_MODULE(DTVDriftCalibration);
DEFINE_ANOTHER_FWK_MODULE(DTVDriftWriter);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(DTFakeTTrigESProducer);
