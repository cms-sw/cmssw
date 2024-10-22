#!/bin/csh

#unsetenv HOME
#echo "HOME" ${HOME}

set runnumber=${1}
set mydate=${2}
set refnumber=${3}
set runNevents=${4}
set CALIB=${5}
set ERA=${6}
set RELEASE=${7}
set SCRAM_ARCH=${8}
set SCRIPT=${9}

set fullSrc0='/eos/cms/store/group/dpg_hcal/comm_hcal/USC'
set fullSrc='NO'
set HistoDir='/eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT/histos'
set WD="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript"

echo ${runnumber} >> ${WD}/LOG/batchlog
grep -q ${runnumber} ${WD}/${CALIB}_LIST/fullSrc0_list_${mydate}
if( ${status} == "0" ) then
set namef0=`grep ${runnumber} ${WD}/${CALIB}_LIST/fullSrc0_list_${mydate}`
set namef=`echo ${namef0} | awk '{print $1}'`
echo ${namef}
if( ${namef} == "run${runnumber}" ) then
set fullSrc=${fullSrc0}/run${runnumber}
else
set fullSrc=${fullSrc0}
endif

echo "here"
endif

echo ${fullSrc} >> ${WD}/LOG/batchlog

if( ${fullSrc} == "NO" ) then
echo "No Batch submission" ${runnumber} >> ${WD}/LOG/batchlog
exit
endif

echo "Batch submission" ${fullSrc} " " ${runnumber} >> ${WD}/LOG/batchlog

###exit

### We are at working node
setenv MH `pwd`

mkdir /tmp/kodolova/${runnumber}
setenv WORK /tmp/kodolova/${runnumber}

cd ${WORK}
source /cvmfs/cms.cern.ch/cmsset_default.csh
cd ${WORK}
cmsrel ${RELEASE} 
cd ${RELEASE}/src
cmsenv

pwd  

ls -l
echo /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/DPGAnalysis

cp -r /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/DPGAnalysis `pwd`/
cp -r /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/Calibration `pwd`/
scramv1 build

#cd ${MH}
#exit

cd ${WORK}
set HistoDirTMP="./"
pwd > ${WD}/LOG/log_${runnumber}

cp ${SCRIPT}/remoteMonitoring_${CALIB}_${ERA}_cfg.py ${WORK}
cp ${WD}/${CALIB}_LIST/runlist.tmp.${2} ${WORK}/runlist.tmp
ls  >> ${WD}/LOG/log_${runnumber}

echo "WORKDIR " ${WD} ${WORK}

echo " Start CMS run " >> ${WD}/LOG/log_${runnumber}
echo ${LD_LIBRARY_PATH} >> ${WD}/LOG/log_${runnumber}
echo ${HistoDir} >> ${WD}/LOG/log_${runnumber} 
echo ${CMSSW_BASE} >> ${WD}/LOG/log_${runnumber}
pwd >> ${WD}/LOG/log_${runnumber}
ls ${WORK} >> ${WD}/LOG/log_${runnumber}

mkdir run${runnumber}

xrdcp ${fullSrc}/USC_${runnumber}.root run${runnumber}/USC_${runnumber}.root 

echo "File was copied to workdir" >> & ${WD}/LOG/log_${runnumber}

set runpath="file:."

cmsRun remoteMonitoring_${CALIB}_${ERA}_cfg.py ${runnumber} ${runpath} ${HistoDirTMP} >> & ${WD}/LOG/log_${runnumber}

ls >> ${WD}/LOG/log_${runnumber}

xrdcp -f ${HistoDirTMP}LED_${runnumber}.root ${HistoDir}/${CALIB}_${runnumber}.root

echo "Output was copied to ${HistoDir}" >> & ${WD}/LOG/log_${runnumber}

###eos ls ${HistoDir}/${CALIB}_${runnumber}.root >> & ${WD}/LOG/log_${runnumber}

echo " After CMS run ">>${WD}/LOG/log_${runnumber}
cd ${MH}
