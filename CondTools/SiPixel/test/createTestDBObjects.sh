#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Pixel CPE DB codes ..."

echo "TESTING Pixel 1D Template DB code ..."
cmsRun ${LOCAL_TEST_DIR}/SiPixelTemplateDBObjectUploader_cfg.py MagField=0.0 Fullname=SiPixelTemplateDBObject_phase1_0T_mc_BoR3_v1_bugfix Map=${LOCAL_TEST_DIR}/../data/phaseI_mapping.csv TemplateFilePath=CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1 || die "Failure running SiPixelTemplateDBObjectUploader_cfg.py" $? 

echo "TESTING Pixel 1D GenErr DB code ..."
cmsRun ${LOCAL_TEST_DIR}/SiPixelGenErrorDBObjectUploader_cfg.py MagField=0.0 Fullname=SiPixelGenErrorDBObject_phase1_0T_mc_BoR3_v1_bugfix Map=${LOCAL_TEST_DIR}/../data/phaseI_mapping.csv GenErrFilePath=CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1 || die "Failure running SiPixelGenErrorDBObjectUploader_cfg.py" $? 

echo "TESTING Pixel 1D Template DB code for Phase-2 ..."
cmsRun ${LOCAL_TEST_DIR}/SiPixelTemplateDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=7 Append=mc_25x100 Map=${LOCAL_TEST_DIR}/../data/phase2_T15_mapping.csv geometry=T15 TemplateFilePath=CalibTracker/SiPixelESProducers/data/Phase2_25by100_templates_2020October || die "Failure running SiPixelTemplateDBObjectUploader_Phase2_cfg.py" $? 

echo "TESTING Pixel 1D GenErr DB code for Phase-2 ..."
cmsRun ${LOCAL_TEST_DIR}/SiPixelGenErrorDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=7 Append=mc_25x100 Map=${LOCAL_TEST_DIR}/../data/phase2_T15_mapping.csv geometry=T15 GenErrFilePath=CalibTracker/SiPixelESProducers/data/Phase2_25by100_templates_2020October || die "Failure running SiPixelGenErrorDBObjectUploader_Phase2_cfg.py" $? 

echo "TESTING Pixel 2D Template DB code for Phase-2 ..."
cmsRun  ${LOCAL_TEST_DIR}/SiPixel2DTemplateDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=7 Append=mc_25x100 Map=${LOCAL_TEST_DIR}/../data/phase2_T15_mapping.csv TemplateFilePath=CalibTracker/SiPixelESProducers/data/Phase2_25by100_templates_2020October denominator=True || die "Failure running SiPixel2DTemplateDBObjectUploader_Phase2_cfg.py" $? 

echo "TESTING SiPixelVCal DB codes ... \n\n"

echo "TESTING Writing SiPixelVCal DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelVCalDB_cfg.py || die "Failure running SiPixelVCalDB_cfg.py" $?

echo "TESTING Reading SiPixelVCal DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelVCalReader_cfg.py || die "Failure running SiPixelVCalReader_cfg.py" $?

echo "TESTING SiPixelLorentzAngle DB codes ... \n\n"

echo "TESTING Writing SiPixelLorentzAngle DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelLorentzAngleDB_cfg.py || die "Failure running SiPixelLorentzAngleDB_cfg.py" $?

echo "TESTING Reading SiPixelLorentzAngle DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelLorentzAngleReader_cfg.py || die "Failure running SiPixelLorentzAngleReader_cfg.py" $?

echo "TESTING SiPixelDynamicInefficiency DB codes ... \n\n"

echo "TESTING Writing SiPixelDynamicInefficiency DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelDynamicInefficiencyDB_cfg.py || die "Failure running SiPixelDynamicInefficiencyDB_cfg.py" $?

echo "TESTING Reading SiPixelDynamicInefficiency DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelDynamicInefficiencyReader_cfg.py || die "Failure running SiPixelDynamicInefficiencyReader_cfg.py" $?

echo "TESTING SiPixelGain Scaling DB codes ... \n\n"

echo "TESTING Writing Scaled SiPixel Gain DB Object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelGainCalibScaler_cfg.py firstRun=278869 maxEvents=12000 || die "Failure running SiPixelGainCalibScaler_cfg.py" $?

echo "TESTING SiPixelQuality DB codes ... \n\n"

echo "TESTING Writing SiPixelQuality DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelBadModuleByHandBuilder_cfg.py || die "Failure running SiPixelBadModuleByHandBuilder_cfg.py" $?

echo "TESTING Reading SiPixelQuality DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelBadModuleReader_cfg.py || die "Failure running SiPixelBadModuleReader_cfg.py" $?
