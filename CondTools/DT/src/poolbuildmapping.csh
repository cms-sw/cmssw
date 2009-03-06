#!/bin/csh
pool_build_object_relational_mapping -f mapping-template-DTReadOutMapping-default.xml -d CondFormatsDTObjectsCapabilities -c oracle://cms_val_lb.cern.ch/CMS_VAL_DT_POOL_OWNER -p val_dt_own_1031 -u CMS_VAL_DT_POOL_OWNER
pool_build_object_relational_mapping -f mapping-template-DTT0-default.xml -d CondFormatsDTObjectsCapabilities -c oracle://cms_val_lb.cern.ch/CMS_VAL_DT_POOL_OWNER -p val_dt_own_1031 -u CMS_VAL_DT_POOL_OWNER
pool_build_object_relational_mapping -f mapping-template-DTTtrig-default.xml -d CondFormatsDTObjectsCapabilities -c oracle://cms_val_lb.cern.ch/CMS_VAL_DT_POOL_OWNER -p val_dt_own_1031 -u CMS_VAL_DT_POOL_OWNER
pool_build_object_relational_mapping -f mapping-template-DTMtime-default.xml -d CondFormatsDTObjectsCapabilities -c oracle://cms_val_lb.cern.ch/CMS_VAL_DT_POOL_OWNER -p val_dt_own_1031 -u CMS_VAL_DT_POOL_OWNER
