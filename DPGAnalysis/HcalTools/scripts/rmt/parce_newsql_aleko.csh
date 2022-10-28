#!/bin/tcsh

### Make list of files: LED, pedestal, laser
#set DAT=`date '+%Y-%m-%d_%H_%M_%S'`
if( ${1} != "LED" && ${1} != "LASER" && ${1} != "PEDESTAL" ) then
echo " Please select run type ./parce_newsql_aleko.csh LED `date '+%Y-%m-%d_%H_%M_%S'` era2019"
exit
endif

set runorigped=286893
set runorigled=286946
set runoriglas=286766

if( ${3} == "era2019" ) then
set runorigled=328157
endif

# ATTENTION:
# for laser and pedestal we put runorig = run current
#

set WD="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript"
set SCRIPT=`pwd`

set LAST=0
set OUTLAST=""
set runref=0
set outfile=""

echo "Here"  ${1}
setenv HCALDQM_DBCONNECT cms_hcl_runinfo/Donald_2_Duck@cms_omds_adg

if( ${1} == "LED" ) then
set LAST=`cat ${WD}/LED_LIST/LASTRUN`
set OUTLAST=${WD}/LED_LIST/LASTRUN
echo ${LAST}
python ${CMSSW_BASE}/src/DPGAnalysis/HcalTools/python/getInfo.py --runmin=${LAST} | grep LED |  awk '{print $1"_"$3"_"$4"_"$7"_"$11}' > tmp.list.${1}
set runref=${runorigled}
touch ${WD}/LED_LIST/runlist.tmp.${2}
set outfile=${WD}/LED_LIST/runlist.tmp.${2}
endif

if( ${1} == "LASER" ) then
echo "Laser"
set runref=${runoriglas}
set LAST=`cat ${WD}/LASER_LIST/LASTRUN`
set OUTLAST=${WD}/LASER_LIST/LASTRUN
python ${CMSSW_BASE}/src/DPGAnalysis/HcalTools/python/getInfo.py --runmin=${LAST} | grep Laser | awk '{print $1"_"$3"_"$4"_"$7"_"$11}' > tmp.list.${1}
touch ${WD}/LASER_LIST/runlist.tmp.${2}
set outfile=${WD}/LASER_LIST/runlist.tmp.${2}
endif

if( ${1} == "PEDESTAL" ) then
set runref=${runorigped}
set LAST=`cat ${WD}/PEDESTAL_LIST/LASTRUN`
set OUTLAST=${WD}/PEDESTAL_LIST/LASTRUN
python ${CMSSW_BASE}/src/DPGAnalysis/HcalTools/python/getInfo.py --runmin=${LAST} | grep Pedestal | awk '{print $1"_"$3"_"$4"_"$7"_"$11}' > tmp.list.${1}
touch ${WD}/PEDESTAL_LIST/runlist.tmp.${2}
set outfile=${WD}/PEDESTAL_LIST/runlist.tmp.${2}
endif

echo "Here 1"
foreach i (`cat tmp.list.${1}`)
set run=`echo ${i} | awk -F_ '{print $1}'`
set mydate=`echo ${i} | awk -F_ '{print $2}'`
set mytime=`echo ${i} | awk -F_ '{print $3}'`
set nevent=`echo ${i} | awk -F_ '{print $4}'`
echo ${run}"_"${run}"_"${mydate}"_"${mytime}"_"${nevent} >> ${outfile}
rm ${OUTLAST}
echo ${run} > ${OUTLAST}
end
if( ${1} == "LED" ) then
eos ls /eos/cms/store/group/dpg_hcal/comm_hcal/USC > ${WD}/LED_LIST/fullSrc0_list_${2}
endif
if( ${1} == "LASER" ) then
eos ls /eos/cms/store/group/dpg_hcal/comm_hcal/USC > ${WD}/LASER_LIST/fullSrc0_list_${2}
endif
if( ${1} == "PEDESTAL" ) then
eos ls /eos/cms/store/group/dpg_hcal/comm_hcal/USC > ${WD}/PEDESTAL_LIST/fullSrc0_list_${2}
endif

