#include "CalibMuon/CSCCalibration/interface/CSCCrosstalkConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCCrosstalkDBConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeCrosstalkConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBCrosstalk.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBGains.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBNoiseMatrix.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBPedestals.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeGainsConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeNoiseMatrixConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakePedestalsConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCGainsConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCGainsDBConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCL1TPParametersConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCNoiseMatrixConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCNoiseMatrixDBConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCPedestalsDBConditions.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_FWK_EVENTSETUP_SOURCE(CSCFakeGainsConditions);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCFakeDBGains);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCFakeCrosstalkConditions);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCFakeDBCrosstalk);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCFakePedestalsConditions);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCFakeDBPedestals);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCFakeNoiseMatrixConditions);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCFakeDBNoiseMatrix);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCCrosstalkConditions);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCCrosstalkDBConditions);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCGainsConditions);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCGainsDBConditions);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCNoiseMatrixConditions);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCNoiseMatrixDBConditions);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCPedestalsDBConditions);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCL1TPParametersConditions);
