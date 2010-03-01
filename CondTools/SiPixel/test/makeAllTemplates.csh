#! /bin/tcsh -f

#set versiontag = 'v2'
set versiontag = $1

cmsRun SiPixelTemplateDBObjectUploader_cfg.py 0 $versiontag
cmsRun SiPixelTemplateDBObjectUploader_cfg.py 2 $versiontag
cmsRun SiPixelTemplateDBObjectUploader_cfg.py 3 $versiontag
cmsRun SiPixelTemplateDBObjectUploader_cfg.py 3.5 $versiontag
cmsRun SiPixelTemplateDBObjectUploader_cfg.py 3.8 $versiontag
cmsRun SiPixelTemplateDBObjectUploader_cfg.py 4 $versiontag

cmsRun SiPixelTemplateDBObjectReader_cfg.py 0 $versiontag
cmsRun SiPixelTemplateDBObjectReader_cfg.py 2 $versiontag
cmsRun SiPixelTemplateDBObjectReader_cfg.py 3 $versiontag
cmsRun SiPixelTemplateDBObjectReader_cfg.py 3.5 $versiontag
cmsRun SiPixelTemplateDBObjectReader_cfg.py 3.8 $versiontag
cmsRun SiPixelTemplateDBObjectReader_cfg.py 4 $versiontag
