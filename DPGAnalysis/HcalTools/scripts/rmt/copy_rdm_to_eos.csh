#!/bin/csh

set WebDir='/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMweb'

ls ${WebDir} | grep LED_ | sort -r > currentlist
foreach i (`cat currentlist`)
ls ${WebDir}/${i}/*.png > /dev/null
if(${status} == "0") then
foreach k (`ls ${WebDir}/${i}/*.html`)
set j=`basename ${k}`
echo ${j}
cat ${k} | sed 's#cms-cpt-software.web.cern.ch\/cms-cpt-software\/General\/Validation\/SVSuite#cms-conddb-dev.cern.ch\/eosweb\/hcal#g'> ${j}.n
cmsStage -f ${j}.n /store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT/${i}/${j}
rm ${j}.n
end
endif
end
rm currentlist
