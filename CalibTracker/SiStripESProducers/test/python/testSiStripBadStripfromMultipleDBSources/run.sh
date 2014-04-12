#!/bin/sh
eval `scramv1 runtime -sh`

cp -f ../dbfile.db .

echo -e "\n Upload n 1"
cmsRun DummyCondDBWriter_SiStripBadModule_fromMultipleDBSources_createTIB_firstIOV_cfg.py
[ $? -ne 0 ] && echo -e "\n Problem in Upload n 1" && exit


echo -e "\n Upload n 2"
cmsRun DummyCondDBWriter_SiStripBadModule_fromMultipleDBSources_createTIB_secondIOV_cfg.py
[ $? -ne 0 ] && echo -e "\n Problem in Upload n 2" && exit


echo -e "\n Upload n 3"
cmsRun DummyCondDBWriter_SiStripBadModule_fromMultipleDBSources_createTID_firstIOV_cfg.py
[ $? -ne 0 ] && echo -e "\n Problem in Upload n 3" && exit


echo -e "\n Upload n 4"
cmsRun DummyCondDBWriter_SiStripBadModule_fromMultipleDBSources_createTID_secondIOV_cfg.py
[ $? -ne 0 ] && echo -e "\n Problem in Upload n 4" && exit


echo -e "\n merge"
cmsRun DummyCondDBWriter_SiStripBadModule_fromMultipleDBSources_merge_cfg.py
[ $? -ne 0 ] && echo -e "\n Problem in merge" && exit


echo -e "\n read"
cmsRun read_DummyCondDBWriter_SiStripQuality_cfg.py
[ $? -ne 0 ] && echo -e "\n Problem in read" && exit

