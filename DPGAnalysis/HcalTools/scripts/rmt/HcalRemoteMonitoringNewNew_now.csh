#!/bin/csh

set runnumber=${1}
set refnumber=${3}
set runNevents=${4}
set CALIB=${5}
set ERA=${6}

set RELEASE=CMSSW_10_4_0

set fullSrc0='/store/group/dpg_hcal/comm_hcal/USC'
set fullSrc1='/store/group/dpg_hcal/comm_hcal/LS1'
set fullSrc='NO'
set HistoDir=`pwd`
set WD=`pwd`

echo ${runnumber} >> ${WD}/LOG/batchlog
grep -q ${runnumber} ${WD}/${CALIB}_LIST/fullSrc0_list_${2}
if( ${status} == "0" ) then
set fullSrc=${fullSrc0}
endif

grep -q ${runnumber} ${WD}/${CALIB}_LIST/fullSrc1_list_${2}
if( ${status} == "0" ) then
set fullSrc=${fullSrc1}
endif

if( ${fullSrc} == "NO" ) then
echo "No Batch submission" ${runnumber} >> ${WD}/batchlog
exit
endif

echo "Batch submission" ${fullSrc} " " ${runnumber} >> ${WD}/batchlog

###exit

### We are at working node
mkdir ${runnumber}
setenv WORK `pwd`/${runnumber}

cd ..
cmsenv
cp ${WD}/remoteMonitoring_${CALIB}_${ERA}_cfg.py ${WORK}/remoteMonitoring_cfg.py
cp ${WD}/RemoteMonitoringMAP.cc ${WORK}
cp ${WD}/compile.csh ${WORK}
cp ${WD}/LogEleMapdb.h ${WORK}
cp ${WD}/${CALIB}_LIST/runlist.tmp.${2} ${WORK}/runlist.tmp

cd ${WORK}

#### cmsRun Start
### Temporarily
#rm LOG/log_${runnumber}
#rm ${HistoDir}/${CALIB}_${runnumber}.root

echo " Start CMS run ">${WD}/log_${runnumber}
echo ${LD_LIBRARY_PATH} >>${WD}/log_${runnumber}

cmsRun remoteMonitoring_cfg.py ${runnumber} ${fullSrc} ${HistoDir} >> & ${WD}/log_${runnumber}
mv ${HistoDir}/LED_${runnumber}.root ${HistoDir}/${CALIB}_${runnumber}.root

echo " After CMS run ">>${WD}/log_${runnumber}
./compile.csh RemoteMonitoringMAP.cc  >> & ${WD}/log_${runnumber}
./RemoteMonitoringMAP.cc.exe "${HistoDir}/${CALIB}_${runnumber}.root" "${HistoDir}/${CALIB}_${refnumber}.root" ${CALIB} >> & ${WD}/log_${runnumber}

