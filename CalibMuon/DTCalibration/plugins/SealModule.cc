#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "CalibMuon/DTCalibration/interface/DTCalibMuonSelection.h"
#include "CalibMuon/DTCalibration/plugins/DTMapGenerator.h"
#include "CalibMuon/DTCalibration/plugins/DTTTrigCalibration.h"
#include "CalibMuon/DTCalibration/plugins/DTTTrigWriter.h"
#include "CalibMuon/DTCalibration/plugins/DTTTrigCorrection.h"
#include "CalibMuon/DTCalibration/plugins/DTTTrigCorrectionFirst.h"
#include "CalibMuon/DTCalibration/plugins/DTT0Calibration.h"
//#include "CalibMuon/DTCalibration/plugins/DTT0CalibrationNew.h"
#include "CalibMuon/DTCalibration/plugins/DTTPDeadWriter.h"
#include "CalibMuon/DTCalibration/plugins/DTT0Correction.h"
#include "CalibMuon/DTCalibration/plugins/DTVDriftCalibration.h"
#include "CalibMuon/DTCalibration/plugins/DTVDriftWriter.h"
#include "CalibMuon/DTCalibration/plugins/DTNoiseComputation.h"
#include "CalibMuon/DTCalibration/plugins/DTNoiseCalibration.h"
#include "CalibMuon/DTCalibration/plugins/DTTTrigOffsetCalibration.h"
#include "CalibMuon/DTCalibration/plugins/DTVDriftSegmentCalibration.h"
#include "CalibMuon/DTCalibration/plugins/DTResidualCalibration.h"

#include "CalibMuon/DTCalibration/plugins/DTFakeT0ESProducer.h"

#include "CalibMuon/DTCalibration/interface/DTTTrigCorrectionFactory.h"
#include "CalibMuon/DTCalibration/plugins/DTTTrigT0SegCorrection.h"
#include "CalibMuon/DTCalibration/plugins/DTTTrigResidualCorrection.h"
#include "CalibMuon/DTCalibration/plugins/DTTTrigMatchRPhi.h"
#include "CalibMuon/DTCalibration/plugins/DTTTrigFillWithAverage.h"
#include "CalibMuon/DTCalibration/plugins/DTTTrigConstantShift.h"

#include "CalibMuon/DTCalibration/interface/DTT0CorrectionFactory.h"
#include "CalibMuon/DTCalibration/plugins/DTT0FillDefaultFromDB.h"
#include "CalibMuon/DTCalibration/plugins/DTT0FillChamberFromDB.h"
#include "CalibMuon/DTCalibration/plugins/DTT0WireInChamberReferenceCorrection.h"
#include "CalibMuon/DTCalibration/plugins/DTT0AbsoluteReferenceCorrection.h"
#include "CalibMuon/DTCalibration/plugins/DTT0FEBPathCorrection.h"

#include "CalibMuon/DTCalibration/interface/DTVDriftPluginFactory.h"
#include "CalibMuon/DTCalibration/plugins/DTVDriftMeanTimer.h"
#include "CalibMuon/DTCalibration/plugins/DTVDriftSegment.h"

DEFINE_FWK_MODULE(DTCalibMuonSelection);
DEFINE_FWK_MODULE(DTMapGenerator);
DEFINE_FWK_MODULE(DTTTrigCalibration);
DEFINE_FWK_MODULE(DTTTrigWriter);
DEFINE_FWK_MODULE(DTTTrigCorrection);
DEFINE_FWK_MODULE(DTTTrigCorrectionFirst);
DEFINE_FWK_MODULE(DTT0Calibration);
//DEFINE_FWK_MODULE(DTT0CalibrationNew);
DEFINE_FWK_MODULE(DTTPDeadWriter);
DEFINE_FWK_MODULE(DTT0Correction);
DEFINE_FWK_MODULE(DTVDriftCalibration);
DEFINE_FWK_MODULE(DTVDriftWriter);
DEFINE_FWK_MODULE(DTNoiseComputation);
DEFINE_FWK_MODULE(DTNoiseCalibration);
DEFINE_FWK_MODULE(DTTTrigOffsetCalibration);
DEFINE_FWK_MODULE(DTVDriftSegmentCalibration);
DEFINE_FWK_MODULE(DTResidualCalibration);

DEFINE_FWK_EVENTSETUP_SOURCE(DTFakeT0ESProducer);

DEFINE_EDM_PLUGIN(DTTTrigCorrectionFactory,dtCalibration::DTTTrigT0SegCorrection,"DTTTrigT0SegCorrection");
DEFINE_EDM_PLUGIN(DTTTrigCorrectionFactory,dtCalibration::DTTTrigResidualCorrection,"DTTTrigResidualCorrection");
DEFINE_EDM_PLUGIN(DTTTrigCorrectionFactory,dtCalibration::DTTTrigMatchRPhi,"DTTTrigMatchRPhi");
DEFINE_EDM_PLUGIN(DTTTrigCorrectionFactory,dtCalibration::DTTTrigFillWithAverage,"DTTTrigFillWithAverage");
DEFINE_EDM_PLUGIN(DTTTrigCorrectionFactory,dtCalibration::DTTTrigConstantShift,"DTTTrigConstantShift");

DEFINE_EDM_PLUGIN(DTT0CorrectionFactory,dtCalibration::DTT0FillDefaultFromDB,"DTT0FillDefaultFromDB");
DEFINE_EDM_PLUGIN(DTT0CorrectionFactory,dtCalibration::DTT0FillChamberFromDB,"DTT0FillChamberFromDB");
DEFINE_EDM_PLUGIN(DTT0CorrectionFactory,dtCalibration::DTT0WireInChamberReferenceCorrection,"DTT0WireInChamberReferenceCorrection");
DEFINE_EDM_PLUGIN(DTT0CorrectionFactory,dtCalibration::DTT0AbsoluteReferenceCorrection,"DTT0AbsoluteReferenceCorrection");
DEFINE_EDM_PLUGIN(DTT0CorrectionFactory,dtCalibration::DTT0FEBPathCorrection, "DTT0FEBPathCorrection");

DEFINE_EDM_PLUGIN(DTVDriftPluginFactory,dtCalibration::DTVDriftMeanTimer,"DTVDriftMeanTimer");
DEFINE_EDM_PLUGIN(DTVDriftPluginFactory,dtCalibration::DTVDriftSegment,"DTVDriftSegment");
