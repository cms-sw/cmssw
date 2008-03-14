#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeGainsConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBGains.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeCrosstalkConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBCrosstalk.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakePedestalsConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBPedestals.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeNoiseMatrixConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBNoiseMatrix.h"
#include "CalibMuon/CSCCalibration/interface/WriteGainsToDB.h"
#include "CalibMuon/CSCCalibration/interface/WriteDBGains.h"
#include "CalibMuon/CSCCalibration/interface/WritePedestalsToDB.h"
#include "CalibMuon/CSCCalibration/interface/WriteDBPedestals.h"
#include "CalibMuon/CSCCalibration/interface/WriteNoiseMatrixToDB.h"
#include "CalibMuon/CSCCalibration/interface/WriteDBNoiseMatrix.h"
#include "CalibMuon/CSCCalibration/interface/WriteCrosstalkToDB.h"
#include "CalibMuon/CSCCalibration/interface/WriteDBCrosstalk.h"
#include "CalibMuon/CSCCalibration/interface/CSCCrosstalkConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCCrosstalkDBConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCGainsConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCGainsDBConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCNoiseMatrixConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCNoiseMatrixDBConditions.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCFakeGainsConditions);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCFakeDBGains);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCFakeCrosstalkConditions);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCFakeDBCrosstalk);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCFakePedestalsConditions);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCFakeDBPedestals);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCFakeNoiseMatrixConditions);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCFakeDBNoiseMatrix);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCCrosstalkConditions);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCCrosstalkDBConditions);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCGainsConditions);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCGainsDBConditions);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCNoiseMatrixConditions);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCNoiseMatrixDBConditions);
DEFINE_ANOTHER_FWK_MODULE(WriteGainsToDB);
DEFINE_ANOTHER_FWK_MODULE(WriteDBGains);
DEFINE_ANOTHER_FWK_MODULE(WritePedestalsToDB);
DEFINE_ANOTHER_FWK_MODULE(WriteDBPedestals);
DEFINE_ANOTHER_FWK_MODULE(WriteNoiseMatrixToDB);
DEFINE_ANOTHER_FWK_MODULE(WriteDBNoiseMatrix);
DEFINE_ANOTHER_FWK_MODULE(WriteCrosstalkToDB);
DEFINE_ANOTHER_FWK_MODULE(WriteDBCrosstalk);
