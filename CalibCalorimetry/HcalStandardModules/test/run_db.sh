#!/bin/sh

export CORAL_AUTH_USER=CMS_COND_HCAL
export CORAL_AUTH_PASSWORD=hcal_cern200603
# export POOL_CATALOG=file:conddbcatalog.xml
export POOL_CATALOG=relationalcatalog_oracle://devdb10/CMS_COND_GENERAL
export CONNECT=oracle://devdb10/${CORAL_AUTH_USER}
echo "Using TNS_ADMIN ${TNS_ADMIN}"

#echo Running cmsRun writeOrcofPedestals.cfg ...
cmsRun sx5.cfg
