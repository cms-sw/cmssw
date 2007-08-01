#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "CalibMuon/DTCalibration/plugins/DTMapGenerator.h"
#include "CalibMuon/DTCalibration/plugins/DTTTrigCalibration.h"
#include "CalibMuon/DTCalibration/plugins/DTTTrigWriter.h"
#include "CalibMuon/DTCalibration/plugins/DTT0Calibration.h"
#include "CalibMuon/DTCalibration/plugins/DTVDriftCalibration.h"
#include "CalibMuon/DTCalibration/plugins/DTVDriftWriter.h"
#include "CalibMuon/DTCalibration/plugins/DTFakeTTrigESProducer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(DTMapGenerator);
DEFINE_ANOTHER_FWK_MODULE(DTTTrigCalibration);
DEFINE_ANOTHER_FWK_MODULE(DTTTrigWriter);
DEFINE_ANOTHER_FWK_MODULE(DTT0Calibration);
DEFINE_ANOTHER_FWK_MODULE(DTVDriftCalibration);
DEFINE_ANOTHER_FWK_MODULE(DTVDriftWriter);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(DTFakeTTrigESProducer);
