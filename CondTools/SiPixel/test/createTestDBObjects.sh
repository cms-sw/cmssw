#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo -e "TESTING Pixel CPE DB codes ..."

echo -e "TESTING Pixel 1D Template DB code ..."
cmsRun ${SCRAM_TEST_PATH}/SiPixelTemplateDBObjectUploader_cfg.py MagField=0.0 Fullname=SiPixelTemplateDBObject_phase1_0T_mc_BoR3_v1_bugfix Map=${SCRAM_TEST_PATH}/../data/phaseI_mapping.csv TemplateFilePath=CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1 || die "Failure running SiPixelTemplateDBObjectUploader_cfg.py" $?

echo -e "TESTING Pixel 1D GenErr DB code ..."
cmsRun ${SCRAM_TEST_PATH}/SiPixelGenErrorDBObjectUploader_cfg.py MagField=0.0 Fullname=SiPixelGenErrorDBObject_phase1_0T_mc_BoR3_v1_bugfix Map=${SCRAM_TEST_PATH}/../data/phaseI_mapping.csv GenErrFilePath=CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1 || die "Failure running SiPixelGenErrorDBObjectUploader_cfg.py" $?

echo -e "TESTING Pixel 1D Template DB code for Phase-2 ..."
cmsRun ${SCRAM_TEST_PATH}/SiPixelTemplateDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=1 Append=mc_25x100_IT615 Map=${SCRAM_TEST_PATH}/../data/phase2_T21_mapping.csv geometry=T21 TemplateFilePath=CalibTracker/SiPixelESProducers/data/Phase2_IT_v6.1.5_25x100_irradiated_v2_mc || die "Failure running SiPixelTemplateDBObjectUploader_Phase2_cfg.py" $?

echo -e "TESTING Pixel 1D GenErr DB code for Phase-2 ..."
cmsRun ${SCRAM_TEST_PATH}/SiPixelGenErrorDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=1 Append=mc_25x100_IT615 Map=${SCRAM_TEST_PATH}/../data/phase2_T21_mapping.csv geometry=T21 GenErrFilePath=CalibTracker/SiPixelESProducers/data/Phase2_IT_v6.1.5_25x100_irradiated_v2_mc || die "Failure running SiPixelGenErrorDBObjectUploader_Phase2_cfg.py" $?

echo -e "TESTING Pixel 2D Template DB code for Phase-2 ..."
cmsRun  ${SCRAM_TEST_PATH}/SiPixel2DTemplateDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=1 Append=mc_25x100_IT615 Map=${SCRAM_TEST_PATH}/../data/phase2_T21_mapping_den.csv TemplateFilePath=CalibTracker/SiPixelESProducers/data/Phase2_IT_v6.1.5_25x100_irradiated_v2_mc denominator=True || die "Failure running SiPixel2DTemplateDBObjectUploader_Phase2_cfg.py" $?

echo -e "TESTING Pixel LorentzAngle DB for Phase-2 ..."
cmsRun  ${SCRAM_TEST_PATH}/SiPixelLorentzAngleDBLoader_Phase2_cfg.py geometry=T25

echo -e "TESTING SiPixelVCal DB codes ... \n\n"

echo -e "TESTING Writing SiPixelVCal DB object ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelVCalDB_cfg.py || die "Failure running SiPixelVCalDB_cfg.py" $?

echo -e "TESTING Reading SiPixelVCal DB object ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelVCalReader_cfg.py || die "Failure running SiPixelVCalReader_cfg.py" $?

echo -e "TESTING SiPixelLorentzAngle DB codes ... \n\n"

echo -e "TESTING Writing SiPixelLorentzAngle DB object ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelLorentzAngleDB_cfg.py || die "Failure running SiPixelLorentzAngleDB_cfg.py" $?

echo -e "TESTING Reading SiPixelLorentzAngle DB object ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelLorentzAngleReader_cfg.py || die "Failure running SiPixelLorentzAngleReader_cfg.py" $?

echo -e "TESTING SiPixelDynamicInefficiency DB codes ... \n\n"

echo -e "TESTING Writing SiPixelDynamicInefficiency DB object ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelDynamicInefficiencyDB_cfg.py || die "Failure running SiPixelDynamicInefficiencyDB_cfg.py" $?

echo -e "TESTING Reading SiPixelDynamicInefficiency DB object ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelDynamicInefficiencyReader_cfg.py || die "Failure running SiPixelDynamicInefficiencyReader_cfg.py" $?

echo -e "TESTING SiPixelGain Scaling DB codes ... \n\n"

echo -e "TESTING Writing Scaled SiPixel Gain DB Object ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelGainCalibScaler_cfg.py firstRun=278869 maxEvents=12000 || die "Failure running SiPixelGainCalibScaler_cfg.py" $?

echo -e "TESTING SiPixelQuality DB codes ... \n\n"

echo -e "TESTING Writing SiPixelQuality DB object ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelBadModuleByHandBuilder_cfg.py || die "Failure running SiPixelBadModuleByHandBuilder_cfg.py" $?

echo -e "TESTING Reading SiPixelQuality DB object ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelBadModuleReader_cfg.py || die "Failure running SiPixelBadModuleReader_cfg.py" $?

echo -e "TESTING Writing SiPixelQuality DB object from ROC list \n\n"

cat <<@EOF >> inputListOfROCs.txt
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 0
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 1
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 2
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 3
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 4
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 5
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 6
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 7
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 8
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 9
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 10
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 11
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 12
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 13
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 14
BPix_BmI_SEC7_LYR3_LDR18F_MOD1_ROC 15
@EOF

echo -e "TESTING Writing SiPixelQuality DB object from input ROC list ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelBadModuleByHandBuilderFromROCList_cfg.py inputROCList=inputListOfROCs.txt || die "Failure running SiPixelBadModuleByHandBuilderFromROCList_cfg.py" $?

if conddb --db SiPixelQualityTest.db list SiPixelQualityTest | grep -q "Run"; then
    echo "Found 'Run' in the output. No error."
else
    echo "Error: 'Run' not found in the output."
    exit 1
fi

echo -e "TESTING Writing SiPixelQuality DB object from input ROC list (by LS IOVs) ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelBadModuleByHandBuilderFromROCList_cfg.py  inputROCList=inputListOfROCs.txt byLumi=True outputTagName=SiPixelQualityTestByLumi || die "Failure running SiPixelBadModuleByHandBuilderFromROCList_cfg.py byLumi=True" $?

if conddb --db SiPixelQualityTest.db list SiPixelQualityTestByLumi | grep -q "Lumi"; then
    echo "Found 'Lumi' in the output. No error."
else
    echo "Error: 'Lumi' not found in the output."
    exit 1
fi

echo -e "TESTING SiPixelQualityProbabilities codes ...\n\n"

echo -e "TESTING Writing SiPixelQualityProbabilities DB object ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelQualityProbabilitiesTestWriter_cfg.py || die "Failure running SiPixelQualityProbabilitiesTestWriter_cfg.py" $?

echo -e "TESTING Reading SiPixelQualityProbabilities DB object ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelQualityProbabilitiesTestReader_cfg.py || die "Failure running SiPixelQualityProbabilitiesTestReader_cfg.py" $?

echo -e "TESTING SiPixelFEDChannelContainer codes ...\n\n"

echo -e "TESTING Writing SiPixelFEDChannelContainer DB object ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/FastSiPixelFEDChannelContainerFromQuality_cfg.py || die "Failure running FastSiPixelFEDChannelContainerFromQuality_cfg.py" $?

echo -e "TESTING Reading SiPixelFEDChannelContainer DB object ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelFEDChannelContainerTestReader_cfg.py || die "Failure running SiPixelFEDChannelContainerTestReader_cfg.py" $?
