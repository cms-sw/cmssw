#!/bin/csh

#unsetenv HOME
#echo "HOME" ${HOME}

set runnumber=${1}
set mydate=${2}
set refnumber=${3}
set runNevents=${4}
set CALIB=${5}
set CALIB1=${5}
if($5 == "MIXED_PEDESTAL" && ${5} == "MIXED_LED" && ${5} == "MIXED_LASER") then
set CALIB1=`echo ${5} | awk -F _ '{print $1}'`
endif
set ERA=${6}
set RELEASE=${7}
set SCRAM_ARCH=${8}
set SCRIPT=${9}
set rundate=${10}
set runtime=${11}

set fullSrc0='/eos/cms/store/group/dpg_hcal/comm_hcal/USC'
set fullSrc='NO'
set HistoDir='/eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT/histos'
set WD="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript"

set WebDir='http://cms-hcal-dpg.web.cern.ch/cms-hcal-dpg/HcalRemoteMonitoring/RMT'
set WDS="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/DPGAnalysis/HcalTools/scripts/rmt"
set WDM="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/DPGAnalysis/HcalTools/macros/rmt"
set PlotsDir='/eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT'

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
mkdir ${runnumber}
setenv WORK `pwd`/${runnumber}
source /cvmfs/cms.cern.ch/cmsset_default.csh
setenv SCRAM_ARCH slc7_amd64_gcc10
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

cd ${WORK}
set HistoDirTMP="./"
pwd > ${WD}/LOG/log_${CALIB}_${runnumber}
echo ${CALIB} >> ${WD}/LOG/log_${CALIB}_${runnumber}
echo ${CALIB1} >> ${WD}/LOG/log_${CALIB}_${runnumber}

cp ${SCRIPT}/remoteMonitoring_${CALIB}_${ERA}_cfg.py ${WORK}
cp ${WD}/${CALIB}_LIST/runlist.tmp.${2} ${WORK}/runlist.tmp
ls  >> ${WD}/LOG/log_${CALIB}_${runnumber}

echo "WORKDIR " ${WD} ${WORK} >> ${WD}/LOG/log_${CALIB}_${runnumber}

echo " Start CMS run " >> ${WD}/LOG/log_${CALIB}_${runnumber}
echo ${LD_LIBRARY_PATH} >> ${WD}/LOG/log_${CALIB}_${runnumber}
echo ${HistoDir} >> ${WD}/LOG/log_${CALIB}_${runnumber} 
echo ${CMSSW_BASE} >> ${WD}/LOG/log_${CALIB}_${runnumber}
pwd >> ${WD}/LOG/log_${CALIB}_${runnumber}
ls ${WORK} >> ${WD}/LOG/log_${CALIB}_${runnumber}

mkdir run${runnumber}

xrdcp ${fullSrc}/USC_${runnumber}.root run${runnumber}/USC_${runnumber}.root 

echo "File was copied to workdir" >> & ${WD}/LOG/log_${CALIB}_${runnumber}

set runpath="file:."

cmsRun remoteMonitoring_${CALIB}_${ERA}_cfg.py ${runnumber} ${runpath} ${HistoDirTMP} >> & ${WD}/LOG/log_${CALIB}_${runnumber}

ls >> ${WD}/LOG/log_${CALIB}_${runnumber}

xrdcp -f ${HistoDirTMP}${CALIB}_${runnumber}.root ${HistoDir}/${CALIB}_${runnumber}.root

echo "Output was copied to ${HistoDir}" >> & ${WD}/LOG/log_${CALIB}_${runnumber}

###eos ls ${HistoDir}/${CALIB}_${runnumber}.root >> & ${WD}/LOG/log_${runnumber}

echo " After CMS run ">>${WD}/LOG/log_${CALIB}_${runnumber}

#### Start root session

ls >> & ${WD}/LOG/logn_${CALIB}_${runnumber}

cp ${WDM}/RemoteMonitoringMAP.cc .
cp ${WDM}/compile.csh .
cp ${WDS}/LogEleMapdb.h .
cp ${WDS}/tmp.list.${CALIB} runlist.tmp

./compile.csh RemoteMonitoringMAP.cc

echo " Start CMS run ">${WD}/LOG/logn_${CALIB}_${runnumber}
echo ${LD_LIBRARY_PATH} >>${WD}/LOG/logn_${CALIB}_${runnumber}

./RemoteMonitoringMAP.cc.exe "${HistoDir}/${CALIB}_${runnumber}.root" "${HistoDir}/${CALIB}_${refnumber}.root" "${CALIB1}" >> ${WD}/LOG/logn_${CALIB}_${runnumber}

ls -l >> ${WD}/LOG/logn_${CALIB}_${runnumber}

#set j=`cat runlist.tmp | grep ${runnumber}`
#echo ${j} >> ${WD}/LOG/batchlog
setenv runtype ${CALIB1}
setenv runHTML NO
#setenv runtime `echo $j | awk -F _ '{print $3}'`
#setenv rundate `echo $j | awk -F _ '{print $2}'`

echo 'RUN Date = '${rundate} ${runtime} >> ${WD}/LOG/batchlog
echo 'RUN Type = '${runtype} >> ${WD}/LOG/batchlog
echo 'Reference RUN number ='${refnumber} >> ${WD}/LOG/batchlog

touch index_draft.html

#adding entry to list of file index_draft.html
set raw=3
echo '<tr>'>> index_draft.html
echo '<td class="s1" align="center">'ktemp'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$runnumber'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$runtype'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$runNevents'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$rundate'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$runtime'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$refnumber'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center"><a href="'$WebDir'/'${CALIB}'_'$runnumber'/MAP.html">'${CALIB}'_'$runnumber'</a></td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">NO</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">OK</td>'>> index_draft.html
echo '</tr>'>> index_draft.html

#### PUT Corresponding calib type to html

ls *.html >> ${WD}/LOG/logn_${CALIB}_${runnumber}

foreach i (`ls *.html`)
cat ${i} | sed s/LED/${CALIB1}/g > ${i}_t
mv ${i}_t ${i}
end

####### Copy to the new site
ls *.png
eos rm -rf ${PlotsDir}/${CALIB1}_${runnumber}
eos mkdir ${PlotsDir}/${CALIB1}_${runnumber} >> & ${WD}/LOG/logn_${CALIB}_${runnumber}

if(${status} == "0") then
#### Copy to the new site
foreach i (`ls *.html`)
xrdcp ${i} ${PlotsDir}/${CALIB1}_${runnumber}
end
foreach k (`ls *.png`)
xrdcp ${k} ${PlotsDir}/${CALIB1}_${runnumber}
end
endif



