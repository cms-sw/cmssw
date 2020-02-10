#!/bin/csh

### set DAT=`date '+%Y-%m-%d_%H_%M_%S'`
set DAT="2014-12-18_11_58_49"
### Get list of done from RDM webpage ###

set WD='/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/CMSSW_10_4_0/src/RecoHcal/HcalPromptAnalysis/test/RDM'

#${WD}/myselect.csh ${DAT}
#ls ${WD}/LED_LIST/runlist.tmp.${DAT}

set jold=194165
foreach i (`cat ${WD}/runlist.test`)
set iold=`echo ${i} | awk -F _ '{print $1}'`
set jold=`echo ${i} | awk -F _ '{print $2}'`
set year=`echo ${i} | awk -F _ '{print $3}' | awk -F - '{print $1}'`
set nevent=`echo ${i} | awk -F _ '{print $5}'`
echo ${iold} ${jold} ${year}
#if( ${year} == "2011" ) then
if( ${nevent} >= "10000") then  
echo  "Start job "
##${WD}/HcalRemoteMonitoringNewt.csh ${iold} ${DAT} ${jold} ${nevent}
${WD}/HcalRemoteMonitoringNewNewRoot.csh ${iold} ${DAT} ${jold}
echo  "End job "
sleep 30
endif
end
