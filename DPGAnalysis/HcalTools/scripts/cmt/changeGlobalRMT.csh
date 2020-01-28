#!/bin/csh
foreach i (`eos ls /eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/GlobalRMT | grep GLOBAL`)
mkdir ${i}
cd ${i}
foreach k (`eos ls /eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/GlobalRMT/${i} | grep html`)
echo ${k}
xrdcp /eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/GlobalRMT/${i}/${k} .
#ls
mv ${k} ${k}.orig
cat ${k}.orig | sed s/conddb-dev/conddb/g > ${k}
rm ${k}.orig
xrdcp -f ${k} /eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/GlobalRMT/${i} 
end
cd ..
rm -rf ${i}
end
