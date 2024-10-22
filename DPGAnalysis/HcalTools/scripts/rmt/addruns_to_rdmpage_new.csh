#!/bin/csh
set DAT=`date '+%Y-%m-%d_%H_%M_%S'`
set RELEASE=${CMSSW_VERSION}
echo ${CMSSW_VERSION} ${RELEASE}
set WebDir='/eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT'
set WD="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/DPGAnalysis/HcalTools/scripts/rmt"

xrdcp ${WebDir}/index.html  index.html.orig.${DAT}
touch ${WD}/index_test.html
set j = 1
foreach i (`cat tmp.list.LED`)
echo ${i}
set RUN=`echo ${i} | awk -F _ '{print $1}'`
echo ${RUN} 
ls ${WebDir}/LED_${RUN}/*.png > /dev/null
if(${status} == "0") then
tail -12 ${WebDir}/LED_${RUN}/index_draft.html > tmp.txt
cat tmp.txt | sed s/ktemp/${j}/ >> ${WD}/index_test.html
@ j = ${j} + "1"
rm tmp.txt
endif
end

set NHEAD=`grep -n 'LED Runs' index.html.orig.${DAT} | awk -F : '{print $1}'`
@ NHEAD = ${NHEAD} + "13"
head -n ${NHEAD} ${WD}/index.html.orig.${DAT} > ${WD}/myindex_head
cat ${WD}/index_test.html >> ${WD}/myindex_head

set NTAIL=`cat ${WD}/index.html.orig.${DAT} | wc -l`

set NLASER=`grep -n 'LASER Runs' index.html.orig.${DAT} | awk -F : '{print $1}'`

set k = ${j}

cat ${WD}/index.html.orig.${DAT} | awk -F \" '2 == /s1/ {print $0}'


#cat ${WD}/index.html.orig.${DAT} > index.html.orig

#foreach i (`cat "index.html.orig"`)

#end

#foreach i (`cat index.html.orig`)
#foreach i (`cat ${WD}/index.html.orig.${DAT}`)
#if(${k} == ${NLASER}) then
#break
#endif
#echo ${i}
#grep s1 ${i}
#if(${status} == "0") then
#set ll=`echo ${i} | cut -c 30 | awk -F \< '{print $1}'`
#echo ${ll} 
#else
#cat ${i} >> ${WD}/myindex_head
#endif
#@ k = ${k} + "1"
#end

#exit
#@ NTAIL = ${NTAIL} + "1"
#tail -n ${NTAIL} ${WD}/index.html.orig.${DAT} >> ${WD}/myindex_tail


