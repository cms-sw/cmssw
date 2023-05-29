#!/bin/csh
set DAT=`date '+%Y-%m-%d_%H_%M_%S'`
set RELEASE=${CMSSW_VERSION}
echo ${CMSSW_VERSION} ${RELEASE}
set WebDir='/eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT'
set WD="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/DPGAnalysis/HcalTools/scripts/rmt"

rm ${WD}/index_test.html
touch ${WD}/index_test.html
xrdcp ${WebDir}/index.html index.html.orig.${DAT}

set NHEAD=`grep -n 'LED Runs' index.html.orig.${DAT} | awk -F : '{print $1}'`
@ NHEAD = ${NHEAD} + "13"
head -n ${NHEAD} ${WD}/index.html.orig.${DAT} >> ${WD}/index_test.html

ls ${WebDir} | grep LED_ | sort -r > ${WD}/mycurrentlist
python mysortled.py ${CMSSW_VERSION}

cp ${WD}/currentlist ${WD}/currentlist.LED
set j=0
foreach i (`cat ${WD}/currentlist`)
ls ${WebDir}/${i}/*.png > /dev/null
if(${status} == "0") then
@ j = ${j} + "1"
tail -12 ${WebDir}/${i}/index_draft.html > tmp0.txt
cat tmp0.txt | sed s/MIXED_MIXED/MIXED/g > tmp.txt
cat tmp.txt | sed s/ktemp/${j}/ >> ${WD}/index_test.html
rm tmp.txt tmp0.txt
endif
end
rm ${WD}/currentlist
rm ${WD}/mycurrentlist

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

ls ${WebDir} | grep PEDESTAL_ | sort -r > ${WD}/mycurrentlist
#### need to set the order according run number
python mysort.py ${CMSSW_VERSION}
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
rm ${WD}/mycurrentlist

cat ${WD}/footer.txt >> ${WD}/index_test.html

cp ${WD}/index_test.html ${WebDir}/index.html
mv ${WD}/index_test.html ${WD}/index_test.html.${DAT}
#rm ${WD}/currentlist
#cat ${WebDir}/index.html | sed 's#cms-cpt-software.web.cern.ch\/cms-cpt-software\/General\/Validation\/SVSuite#cms-conddb-prod.cern.ch\/eosweb\/hcal#g'> tmp.html
#xrdcp -f tmp.html /eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT/index.html
#rm tmp.html
