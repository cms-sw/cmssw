#!/bin/csh

set RELEASE=${CMSSW_VERSION}
### Get list of done from RDM webpage ###
set TYPE=${1}
echo ${TYPE}
if( ${TYPE} != "LED" && ${TYPE} != "LASER" && ${TYPE} != "PEDESTAL" ) then
echo "Please check type " ${TYPE} "should be LED or LASER or PEDESTAL"
exit
endif

set WD="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript"
set SCRIPT=`pwd`
set PYTHON=${CMSSW_BASE}/src/DPGAnalysis/HcalTools/python
echo ${WD}

set jold=194165
foreach i (`cat tmp.list.${TYPE}`)
echo "Run" ${i}
set iold=`echo ${i} | awk -F _ '{print $1}'`
set date=`echo ${i} | awk -F _ '{print $2}'`
set time=`echo ${i} | awk -F _ '{print $3}'`
set year=`echo ${date} | awk -F - '{print $1}'`
set nevent=`echo ${i} | awk -F _ '{print $4}'`
echo ${iold} ${jold} ${year} ${nevent}

echo "Make a job"
${SCRIPT}/HcalRemoteMonitoringNewNewRoot.csh ${TYPE} ${iold} ${iold} ${nevent} ${RELEASE}
echo  "End job "
sleep 1
end
