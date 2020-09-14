cmsRun SiPixelTemplateDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=6 Append=mc_25x100 Map=../data/phase2_T15_mapping.csv geometry=T15 TemplateFilePath=CondTools/SiPixel/data/25by100_dual_slope_temps

cmsRun SiPixelGenErrorDBObjectUploader_Phase2_cfg.py MagField=3.8 Version=6 Append=mc_25x100 Map=../data/phase2_T15_mapping.csv geometry=T15 GenErrFilePath=CondTools/SiPixel/data/25by100_dual_slope_temps

