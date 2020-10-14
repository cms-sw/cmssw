function die { echo $1: status $2 ; exit $2; }

echo "TESTING Pixel CPE DB codes (for Phase-2) ..."

cmsRun  ${LOCAL_TEST_DIR}/SiPixelTemplateDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=6 Append=mc_25x100 Map= ${LOCAL_TEST_DIR}/../data/phase2_T15_mapping.csv geometry=T15 TemplateFilePath= ${LOCAL_TEST_DIR}/CondTools/SiPixel/data/25by100_dual_slope_temps || die "Failure running SiPixelTemplateDBObjectUploader_cfg.py" $? 

cmsRun  ${LOCAL_TEST_DIR}/SiPixelGenErrorDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=6 Append=mc_25x100 Map= ${LOCAL_TEST_DIR}/../data/phase2_T15_mapping.csv geometry=T15 GenErrFilePath= ${LOCAL_TEST_DIR}/CondTools/SiPixel/data/25by100_dual_slope_temps || die "Failure running SiPixelGenErrorDBObjectUploader_Phase2_cfg.py" $? 

#For 2D templates 
cmsRun  ${LOCAL_TEST_DIR}/SiPixel2DTemplateDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=7 Append=mc_25x100 Map= ${LOCAL_TEST_DIR}/../data/Phase2_25by100_templates_2020October/phase2_T15_mapping.csv TemplateFilePath= ${LOCAL_TEST_DIR}/CondTools/SiPixel/data/Phase2_25by100_templates_2020October denominator=True || die "Failure running SiPixel2DTemplateDBObjectUploader_Phase2_cfg.py" $? 