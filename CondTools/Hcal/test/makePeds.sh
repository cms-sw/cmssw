#!/bin/sh

export CORAL_AUTH_USER=CMS_COND_HCAL
export CORAL_AUTH_PASSWORD=hcal_cern200603
export POOL_CATALOG=file:conddbcatalog.xml
export CONNECT=oracle://devdb10/${CORAL_AUTH_USER}
echo "Using TNS_ADMIN ${TNS_ADMIN}"

echo Running hcalCalibrationsCopy $*
hcalCalibrationsCopy $*
#hcalCalibrationsCopy pedestals -input CMS_HCL_PRTTYPE_HCAL_READER/HCALGenericReader@CMSOMDS -inputtag 'TAG=test_hcal_pedestals_v1 RUN=1' $*

# hcalCalibrationsCopy pedestals -input CMS_HCL_PRTTYPE_HCAL_READER/HCALGenericReader@CMSOMDS -inputtag 'TAG=test_hcal_pedestals_v1 RUN=1' -output oracle://cms_val_lb.cern.ch/CMS_VAL_HCAL_POOL_OWNER -outputrun 4 -outputtag test_hcal_pedestals_v01
#gdb hcalCalibrationsCopy 
