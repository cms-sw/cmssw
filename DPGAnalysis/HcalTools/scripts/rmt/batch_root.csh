#!/bin/csh

### set DAT=`date '+%Y-%m-%d_%H_%M_%S'`
set DAT="2015-09-15_18_50_34"
set RELEASE=CMSSW_10_4_0
### Get list of done from RDM webpage ###
set TYPE=${1}
echo ${TYPE}
if( ${TYPE} != "LED" && ${TYPE} != "LASER" && ${TYPE} != "PEDESTAL" ) then
echo "Please check type " ${TYPE} "should be LED or LASER or PEDESTAL"
exit
endif

set WD="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/RecoHcal/HcalPromptAnalysis/test/RDM"

cd $SCRATCH

ls ${WD}/${TYPE}_LIST/runlist.tmp.${DAT}
touch rootlog
set jold=194165
foreach i (`cat ${WD}/${TYPE}_LIST/runlist.tmp.${DAT}`)
set iold=`echo ${i} | awk -F _ '{print $1}'`
set jold=`echo ${i} | awk -F _ '{print $2}'`
set year=`echo ${i} | awk -F _ '{print $3}' | awk -F - '{print $1}'`
set nevent=`echo ${i} | awk -F _ '{print $5}'`
echo ${iold} ${jold} ${year} >> rootlog
#if( ${year} == "2011" ) then
if( ${nevent} >= "500") then  
echo  "Start job " >> rootlog
${WD}/HcalRemoteMonitoringNewNewRoot.csh ${iold} ${DAT} ${jold} ${nevent} ${TYPE} >> &rootlog
echo  "End job "
endif
end
