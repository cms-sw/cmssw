#!/bin/tcsh

### Make list of files: LED, pedestal, laser
#set DAT=`date '+%Y-%m-%d_%H_%M_%S'`
if( ${1} != "LED" && ${1} != "LASER" && ${1} != "PEDESTAL" && ${1} != "MIXED_PEDESTAL" && ${1} != "MIXED_LED" && ${1} != "MIXED_LASER" ) then
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

set LAST=`cat ${WD}/${1}_LIST/LASTRUN`

set OUTLAST=${WD}/${1}_LIST/LASTRUN

set outfile=${WD}/${1}_LIST/runlist.tmp.${2}
echo ${LAST} ${OUTLAST} ${outfile}

if($1 == "MIXED_PEDESTAL" || ${1} == "MIXED_LED" || ${1} == "MIXED_LASER") then
python ${CMSSW_BASE}/src/DPGAnalysis/HcalTools/python/getInfo.py --runmin=${LAST} | grep led-ped-Gsel-bv-sequence |  awk '{print $1"_"$3"_"$4"_"$7"_"$11}' > tmp.list.${1}
set runref=${runorigled}
endif
if( ${1} == "LED") then
python ${CMSSW_BASE}/src/DPGAnalysis/HcalTools/python/getInfo.py --runmin=${LAST} | grep LED |  awk '{print $1"_"$3"_"$4"_"$7"_"$11}' > tmp.list.${1}
set runref=${runorigled}
endif
if( ${1} == "LASER" ) then
python ${CMSSW_BASE}/src/DPGAnalysis/HcalTools/python/getInfo.py --runmin=${LAST} | grep Laser | awk '{print $1"_"$3"_"$4"_"$7"_"$11}' > tmp.list.${1}
set runref=${runoriglas}
endif
if( ${1} == "PEDESTAL" ) then
set runref=${runorigped}
python ${CMSSW_BASE}/src/DPGAnalysis/HcalTools/python/getInfo.py --runmin=${LAST} | grep Pedestal | awk '{print $1"_"$3"_"$4"_"$7"_"$11}' > tmp.list.${1}
endif

touch ${outfile}

foreach i (`cat tmp.list.${1}`)
echo ${i}
set run=`echo ${i} | awk -F_ '{print $1}'`
set mydate=`echo ${i} | awk -F_ '{print $2}'`
set mytime=`echo ${i} | awk -F_ '{print $3}'`
set nevent=`echo ${i} | awk -F_ '{print $4}'`
echo ${outfile}
echo ${run}"_"${run}"_"${mydate}"_"${mytime}"_"${nevent} >> ${outfile}
echo ${OUTLAST}
echo ${run} > ${OUTLAST}
end

eos ls /eos/cms/store/group/dpg_hcal/comm_hcal/USC > ${WD}/${1}_LIST/fullSrc0_list_${2}
