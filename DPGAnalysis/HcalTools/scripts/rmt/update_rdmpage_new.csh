#!/bin/csh

set WebDir='/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMweb'
set WD='/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/CMSSW_10_4_0/src/RecoHcal/HcalPromptAnalysis/test/RDM'

touch ${WD}/index_test.html
cat ${WD}/header_led.txt >> ${WD}/index_test.html

ls ${WebDir} | grep LED_ | sort -r > ${WD}/currentlist
set j=0
foreach i (`cat ${WD}/currentlist`)
ls ${WebDir}/${i}/*.png > /dev/null
if(${status} == "0") then
@ j = ${j} + "1"
tail -12 ${WebDir}/${i}/index_draft.html > tmp.txt
cat tmp.txt | sed s/ktemp/${j}/ >> ${WD}/index_test.html
rm tmp.txt
endif
end
rm ${WD}/currentlist

cat ${WD}/header_laser.txt >> ${WD}/index_test.html
ls ${WebDir} | grep LASER_ | sort -r > ${WD}/currentlist

set j=0
foreach i (`cat ${WD}/currentlist`)
ls ${WebDir}/${i}/*.png > /dev/null
if(${status} == "0") then
@ j = ${j} + "1"
tail -12 ${WebDir}/${i}/index_draft.html > tmp.txt
cat tmp.txt | sed s/ktemp/${j}/ >> ${WD}/index_test.html
rm tmp.txt
endif
end
rm ${WD}/currentlist

cat ${WD}/header_pedestal.txt >> ${WD}/index_test.html
ls ${WebDir} | grep PEDESTAL_ | sort -r > ${WD}/currentlist

set j=0
foreach i (`cat ${WD}/currentlist`)
ls ${WebDir}/${i}/*.png > /dev/null
if(${status} == "0") then
@ j = ${j} + "1"
tail -12 ${WebDir}/${i}/index_draft.html > tmp.txt
cat tmp.txt | sed s/ktemp/${j}/ >> ${WD}/index_test.html
rm tmp.txt
endif
end
set DAT=`date '+%Y-%m-%d_%H_%M_%S'`
cat ${WD}/footer.txt >> ${WD}/index_test.html
mv ${WebDir}/index.html ${WebDir}/index.html.orig.${DAT}
mv ${WD}/index_test.html ${WebDir}/index.html
rm ${WD}/currentlist
cat ${WebDir}/index.html | sed 's#cms-cpt-software.web.cern.ch\/cms-cpt-software\/General\/Validation\/SVSuite#cms-conddb-prod.cern.ch\/eosweb\/hcal#g'> tmp.html

xrdcp -f tmp.html /eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT/index.html
rm tmp.html
