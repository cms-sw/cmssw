#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo -e "TESTING Pixel CPE DB codes ..."

echo -e "TESTING Pixel 1D Template DB code ..."
cmsRun ${LOCAL_TEST_DIR}/SiPixelTemplateDBObjectUploader_cfg.py MagField=0.0 Fullname=SiPixelTemplateDBObject_phase1_0T_mc_BoR3_v1_bugfix Map=${LOCAL_TEST_DIR}/../data/phaseI_mapping.csv TemplateFilePath=CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1 || die "Failure running SiPixelTemplateDBObjectUploader_cfg.py" $? 

echo -e "TESTING Pixel 1D GenErr DB code ..."
cmsRun ${LOCAL_TEST_DIR}/SiPixelGenErrorDBObjectUploader_cfg.py MagField=0.0 Fullname=SiPixelGenErrorDBObject_phase1_0T_mc_BoR3_v1_bugfix Map=${LOCAL_TEST_DIR}/../data/phaseI_mapping.csv GenErrFilePath=CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1 || die "Failure running SiPixelGenErrorDBObjectUploader_cfg.py" $? 

echo -e "TESTING Pixel 1D Template DB code for Phase-2 ..."
cmsRun ${LOCAL_TEST_DIR}/SiPixelTemplateDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=1 Append=mc_25x100_IT615 Map=${LOCAL_TEST_DIR}/../data/phase2_T21_mapping.csv geometry=T21 TemplateFilePath=CalibTracker/SiPixelESProducers/data/Phase2_IT_v6.1.5_25x100_irradiated_v2_mc || die "Failure running SiPixelTemplateDBObjectUploader_Phase2_cfg.py" $? 

echo -e "TESTING Pixel 1D GenErr DB code for Phase-2 ..."
cmsRun ${LOCAL_TEST_DIR}/SiPixelGenErrorDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=1 Append=mc_25x100_IT615 Map=${LOCAL_TEST_DIR}/../data/phase2_T21_mapping.csv geometry=T21 GenErrFilePath=CalibTracker/SiPixelESProducers/data/Phase2_IT_v6.1.5_25x100_irradiated_v2_mc || die "Failure running SiPixelGenErrorDBObjectUploader_Phase2_cfg.py" $? 

echo -e "TESTING Pixel 2D Template DB code for Phase-2 ..."
cmsRun  ${LOCAL_TEST_DIR}/SiPixel2DTemplateDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=1 Append=mc_25x100_IT615 Map=${LOCAL_TEST_DIR}/../data/phase2_T21_mapping_den.csv TemplateFilePath=CalibTracker/SiPixelESProducers/data/Phase2_IT_v6.1.5_25x100_irradiated_v2_mc denominator=True || die "Failure running SiPixel2DTemplateDBObjectUploader_Phase2_cfg.py" $? 

echo -e "TESTING SiPixelVCal DB codes ... \n\n"

echo -e "TESTING Writing SiPixelVCal DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelVCalDB_cfg.py || die "Failure running SiPixelVCalDB_cfg.py" $?

echo -e "TESTING Reading SiPixelVCal DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelVCalReader_cfg.py || die "Failure running SiPixelVCalReader_cfg.py" $?

echo -e "TESTING SiPixelLorentzAngle DB codes ... \n\n"

echo -e "TESTING Writing SiPixelLorentzAngle DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelLorentzAngleDB_cfg.py || die "Failure running SiPixelLorentzAngleDB_cfg.py" $?

echo -e "TESTING Reading SiPixelLorentzAngle DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelLorentzAngleReader_cfg.py || die "Failure running SiPixelLorentzAngleReader_cfg.py" $?

echo -e "TESTING SiPixelDynamicInefficiency DB codes ... \n\n"

echo -e "TESTING Writing SiPixelDynamicInefficiency DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelDynamicInefficiencyDB_cfg.py || die "Failure running SiPixelDynamicInefficiencyDB_cfg.py" $?

echo -e "TESTING Reading SiPixelDynamicInefficiency DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelDynamicInefficiencyReader_cfg.py || die "Failure running SiPixelDynamicInefficiencyReader_cfg.py" $?

echo -e "TESTING SiPixelGain Scaling DB codes ... \n\n"

echo -e "TESTING Writing Scaled SiPixel Gain DB Object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelGainCalibScaler_cfg.py firstRun=278869 maxEvents=12000 || die "Failure running SiPixelGainCalibScaler_cfg.py" $?

echo -e "TESTING SiPixelQuality DB codes ... \n\n"

echo -e "TESTING Writing SiPixelQuality DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelBadModuleByHandBuilder_cfg.py || die "Failure running SiPixelBadModuleByHandBuilder_cfg.py" $?

echo -e "TESTING Reading SiPixelQuality DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelBadModuleReader_cfg.py || die "Failure running SiPixelBadModuleReader_cfg.py" $?

echo -e "TESTING SiPixelQualityProbabilities codes ...\n\n"

echo -e "TESTING Writing SiPixelQualityProbabilities DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelQualityProbabilitiesTestWriter_cfg.py || die "Failure running SiPixelQualityProbabilitiesTestWriter_cfg.py" $?

echo -e "TESTING Reading SiPixelQualityProbabilities DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelQualityProbabilitiesTestReader_cfg.py || die "Failure running SiPixelQualityProbabilitiesTestReader_cfg.py" $?

echo -e "TESTING SiPixelFEDChannelContainer codes ...\n\n"

echo -e "TESTING Writing SiPixelFEDChannelContainer DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/FastSiPixelFEDChannelContainerFromQuality_cfg.py || die "Failure running FastSiPixelFEDChannelContainerFromQuality_cfg.py" $?

echo -e "TESTING Reading SiPixelFEDChannelContainer DB object ...\n\n"
cmsRun  ${LOCAL_TEST_DIR}/SiPixelFEDChannelContainerTestReader_cfg.py || die "Failure running SiPixelFEDChannelContainerTestReader_cfg.py" $?
