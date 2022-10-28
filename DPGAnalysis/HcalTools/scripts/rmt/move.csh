#!/bin/tcsh
foreach i (`cat tmp.list.LED`)
set j=`echo ${i} | awk -F _ '{print $1}'`
foreach jj (`ls /eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT/LED_${j} | grep html`)
set k=`basename ${jj} .n`
echo ${k}_${jj}
xrdcp /eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT/LED_${j}/${jj} .
mv ${jj} ${k}
xrdcp ${k} /eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT/LED_${j}
end
end
