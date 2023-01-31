#!/bin/csh

###set DAT="TEST"
###set DAT="2015-10-07_14_50_47"

cmsenv

set DAT=`date '+%Y-%m-%d_%H_%M_%S'`
set ERA=${2}
set RELEASE=${CMSSW_VERSION}

### Get list of done from RDM webpage ###
set TYPE=${1}
echo ${TYPE}
if( ${TYPE} != "LED" && ${TYPE} != "LASER" && ${TYPE} != "PEDESTAL" && ${TYPE} != "MIXED_PEDESTAL" && ${TYPE} != "MIXED_LED" && ${TYPE} != "MIXED_LASER") then
echo "Please check type " ${TYPE} "should be LED or LASER or PEDESTAL or MIXED_PEDESTAL or MIXED_LED or MIXED_LASER"
exit
endif

set WD="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript"
set SCRIPT=`pwd`
set PYTHON=${CMSSW_BASE}/src/DPGAnalysis/HcalTools/python
echo ${WD}

${CMSSW_BASE}/src/DPGAnalysis/HcalTools/scripts/rmt/parce_newsql_aleko.csh ${TYPE} ${DAT} ${ERA} 

ls ${WD}/${TYPE}_LIST/runlist.tmp.${DAT}
rm ${WD}/${TYPE}_LIST/RUNS_TMP
cat ${WD}/${TYPE}_LIST/runlist.tmp.${DAT} > ${WD}/${TYPE}_LIST/RUNS_TMP

set jold=194165
foreach i (`cat ${WD}/${TYPE}_LIST/RUNS_TMP`)
echo "Run" ${i}

set iold=`echo ${i} | awk -F _ '{print $1}'`
set jold=`echo ${i} | awk -F _ '{print $2}'`
set rundat = `echo ${i} | awk -F _ '{print $3}'`
set runtime = `echo ${i} | awk -F _ '{print $4}'`
set year=`echo ${i} | awk -F _ '{print $3}' | awk -F - '{print $1}'`
set nevent=`echo ${i} | awk -F _ '{print $5}'`
echo ${iold} ${jold} ${year} ${nevent}
if( ${nevent} != "None" ) then
if( ${nevent} >= "500" && ${nevent} <= "11000") then  
echo  "Start job "

touch ${WD}/BATCHLOG/batchlog_${TYPE}_${iold}.log ${WD}/BATCHLOG/ebatchlog_${TYPE}_${iold}.log
set MYOUTJOB=${WD}/BATCHLOG/batchlog_${TYPE}_${iold}.log 
echo ${MYOUTJOB}
set MYERRJOB=${WD}/BATCHLOG/ebatchlog_${TYPE}_${iold}.log
echo ${MYERRJOB}
set MYCONDOR=${WD}/BATCHLOG/condor.log
echo ${MYCONDOR}

rm ${WD}/BSUB/job_${TYPE}_${iold}_${jold}.sub

cat >> ${WD}/BSUB/job_${TYPE}_${iold}_${jold}.sub << EOF
executable            = ${SCRIPT}/HcalRemoteMonitoringNewNewCombine.csh
arguments             = ${iold} ${DAT} ${iold} ${nevent} ${TYPE} ${ERA} ${RELEASE} ${SCRAM_ARCH} ${PYTHON} ${rundat} ${runtime}
output                = ${MYOUTJOB}
error                 = ${MYERRJOB}
log                   = ${MYCONDOR}
+JobFlavour           = "workday"
queue 1
EOF

echo "Make a job"
condor_submit ${WD}/BSUB/job_${TYPE}_${iold}_${jold}.sub
echo  "End job "
sleep 1
endif
endif
end
