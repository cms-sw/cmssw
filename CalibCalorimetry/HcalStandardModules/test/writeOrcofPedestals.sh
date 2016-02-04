#!/bin/sh

export CORAL_AUTH_USER=CMS_COND_HCAL
export CORAL_AUTH_PASSWORD=hcal_cern200603
export POOL_CATALOG=file:conddbcatalog.xml
export CONNECT=oracle://devdb10/${CORAL_AUTH_USER}
echo "Using TNS_ADMIN ${TNS_ADMIN}"

echo Running cmsRun writeOrcofPedestals.cfg ...
cmsRun writeOrcofPedestals.cfg
