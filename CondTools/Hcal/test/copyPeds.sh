#!/bin/sh

export POOL_AUTH_USER=CMS_VAL_HCAL_POOL_OWNER
export POOL_AUTH_PASSWORD=val_hca_own_1031
#export POOL_OUTMSG_LEVEL=debug

export TNS_ADMIN=$HOME
export POOL_CATALOG=xmlcatalog_file:PoolFileCatalog.xml

echo Running hcalCalibrationsCopy $*
hcalCalibrationsCopy $*
#hcalCalibrationsCopy pedestals -input CMS_HCL_PRTTYPE_HCAL_READER/HCALGenericReader@CMSOMDS -inputtag 'TAG=test_hcal_pedestals_v1 RUN=1' $*

# hcalCalibrationsCopy pedestals -input CMS_HCL_PRTTYPE_HCAL_READER/HCALGenericReader@CMSOMDS -inputtag 'TAG=test_hcal_pedestals_v1 RUN=1' -output oracle://cms_val_lb.cern.ch/CMS_VAL_HCAL_POOL_OWNER -outputrun 4 -outputtag test_hcal_pedestals_v01
#gdb hcalCalibrationsCopy 
