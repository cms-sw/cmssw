#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeGainsConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeCrosstalkConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakePedestalsConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeNoiseMatrixConditions.h"
#include "CalibMuon/CSCCalibration/interface/WriteGainsToDB.h"
#include "CalibMuon/CSCCalibration/interface/WritePedestalsToDB.h"
#include "CalibMuon/CSCCalibration/interface/WriteNoiseMatrixToDB.h"
#include "CalibMuon/CSCCalibration/interface/WriteCrosstalkToDB.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCFakeGainsConditions);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCFakeCrosstalkConditions);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCFakePedestalsConditions);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCFakeNoiseMatrixConditions);
DEFINE_ANOTHER_FWK_MODULE(WriteGainsToDB);
DEFINE_ANOTHER_FWK_MODULE(WritePedestalsToDB);
DEFINE_ANOTHER_FWK_MODULE(WriteNoiseMatrixToDB);
DEFINE_ANOTHER_FWK_MODULE(WriteCrosstalkToDB);
