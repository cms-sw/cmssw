#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Pixel CPE DB codes ..."
cmsRun ${LOCAL_TEST_DIR}/SiPixelTemplateDBObjectUploader_cfg.py Fullname=SiPixelTemplateDBObject_phase1_0T_mc_BoR3_v1_bugfix Map=${LOCAL_TEST_DIR}/../data/phaseI_mapping.csv TemplateFilePath=CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1 || die "Failure running SiPixelTemplateDBObjectUploader_cfg.py" $? 
cmsRun ${LOCAL_TEST_DIR}/SiPixelGenErrorDBObjectUploader_cfg.py Fullname=SiPixelGenErrorDBObject_phase1_0T_mc_BoR3_v1_bugfix Map=${LOCAL_TEST_DIR}/../data/phaseI_mapping.csv GenErrFilePath=CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1 || die "Failure running SiPixelGenErrorDBObjectUploader_cfg.py" $? 
