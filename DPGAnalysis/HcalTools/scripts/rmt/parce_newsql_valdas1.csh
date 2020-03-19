#!/bin/tcsh

### Make list of files: LED, pedestal, laser
#set DAT=`date '+%Y-%m-%d_%H_%M_%S'`
if( ${1} != "LED" && ${1} != "laser" && ${1} != "pedestal" ) then
echo " Please select run type ./parce_newsql_valdas.csh LED `date '+%Y-%m-%d_%H_%M_%S'`"
exit
endif

set runorigped=286893
set runorigled=286946
set runoriglas=286766
# ATTENTION:
# for laser and pedestal we put runorig = run current
#
set RELEASE=CMSSW_10_4_0
set WD=/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/RecoHcal/HcalPromptAnalysis/test/RDM

set LAST=0
set OUTLAST=lasttmp
set runref=0
set outfile=tmp

echo "Here 0"

if( ${1} == "LED" ) then
set LAST=`cat ${WD}/LED_LIST/LASTRUN1`
set OUTLAST=${WD}/LED_LIST/LASTRUN1
python /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/RecoHcal/HcalPromptAnalysis/test/RDM/hcal_runs_valdas.py ${LAST} - | grep ${1} | sed 's/"//g' > tmp.list 
cat tmp.list | sed s/" "/_/g > tmp.list.${1}
rm tmp.list
set runref=${runorigled}
touch ${WD}/LED_LIST/runlist.tmp.${2}
set outfile=${WD}/LED_LIST/runlist.tmp.${2}
endif

if( ${1} == "laser" ) then
echo "Laser"
set runref=${runoriglas}
set LAST=`cat ${WD}/LASER_LIST/LASTRUN1`
set OUTLAST=${WD}/LASER_LIST/LASTRUN1
python /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/RecoHcal/HcalPromptAnalysis/test/RDM/hcal_runs_valdas.py ${LAST} - | grep ${1} | sed 's/"//g'  > tmp.list
cat tmp.list | sed s/" "/_/g > tmp.list.${1}
rm tmp.list
touch ${WD}/LASER_LIST/runlist.tmp.${2}
set outfile=${WD}/LASER_LIST/runlist.tmp.${2}
endif

if( ${1} == "pedestal" ) then
set runref=${runorigped}
set LAST=`cat ${WD}/PEDESTAL_LIST/LASTRUN1`
set OUTLAST=${WD}/PEDESTAL_LIST/LASTRUN1
python /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/RecoHcal/HcalPromptAnalysis/test/RDM/hcal_runs_valdas.py ${LAST} - | grep ${1} | sed 's/"//g' > tmp.list
cat tmp.list | sed s/" "/_/g > tmp.list.${1}
rm tmp.list
touch ${WD}/PEDESTAL_LIST/runlist.tmp.${2}
set outfile=${WD}/PEDESTAL_LIST/runlist.tmp.${2}
endif
echo "Here 1"
foreach i (`cat tmp.list.${1}`)
set run=`echo ${i} | awk -F_ '{print $1}'`
set mydate=`echo ${i} | awk -F_ '{print $3}'`
set mytime=`echo ${i} | awk -F_ '{print $4}'`
set nevent=`echo ${i} | awk -F_ '{print $5}'`
if( ${run} > ${LAST} ) then
if( ${1} == "LED" ) then
echo ${run}"_"${runref}"_"${mydate}"_"${mytime}"_"${nevent} >> ${outfile}
else
echo ${run}"_"${run}"_"${mydate}"_"${mytime}"_"${nevent} >> ${outfile}
endif
rm ${OUTLAST}
echo ${run} > ${OUTLAST}
endif
end
if( ${1} == "LED" ) then
eos ls /eos/cms/store/group/dpg_hcal/comm_hcal/USC > ${WD}/LED_LIST/fullSrc0_list_${2}
endif
if( ${1} == "laser" ) then
eos ls /eos/cms/store/group/dpg_hcal/comm_hcal/USC > ${WD}/LASER_LIST/fullSrc0_list_${2}
endif
if( ${1} == "pedestal" ) then
eos ls /eos/cms/store/group/dpg_hcal/comm_hcal/USC > ${WD}/PEDESTAL_LIST/fullSrc0_list_${2}
endif

