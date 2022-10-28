#!/bin/csh
pwd
set CALIB=${1}
set runnumber=${2}
set refnumber=${3}
set runNevents=${4}
set RELEASE=${5}

set WebDir='http://cms-hcal-dpg.web.cern.ch/cms-hcal-dpg/HcalRemoteMonitoring/RMT'
set WD="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript"
set WDS="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/DPGAnalysis/HcalTools/scripts/rmt"
set WDM="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/DPGAnalysis/HcalTools/macros/rmt"
set HistoDir='/eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT/histos'
set PlotsDir='/eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT' 

##### At local WN
mkdir ${runnumber}
setenv WORK `pwd`/${runnumber}
source /cvmfs/cms.cern.ch/cmsset_default.csh
setenv SCRAM_ARCH slc7_amd64_gcc900
cd ${WORK}
cmsrel ${RELEASE}
cd ${RELEASE}/src
cmsenv
cd ${WORK}

xrdcp -f ${HistoDir}/LED_${runnumber}.root .

ls >> & ${WD}/LOG/logn_${runnumber}

cp ${WDM}/RemoteMonitoringMAP.cc .
cp ${WDM}/compile.csh .
cp ${WDS}/LogEleMapdb.h .
cp ${WDS}/tmp.list.LED runlist.tmp

./compile.csh RemoteMonitoringMAP.cc

echo " Start CMS run ">${WD}/LOG/logn_${runnumber}
echo ${LD_LIBRARY_PATH} >>${WD}/LOG/logn_${runnumber}

./RemoteMonitoringMAP.cc.exe "${HistoDir}/${CALIB}_${runnumber}.root" "${HistoDir}/${CALIB}_${refnumber}.root" "${CALIB}" >> ${WD}/LOG/logn_${runnumber}

ls -l >> ${WD}/LOG/log_${runnumber}

set j=`cat runlist.tmp | grep ${runnumber}`
echo ${j} >> ${WD}/LOG/batchlog    
setenv runtype ${CALIB}
setenv runHTML NO
setenv runtime `echo $j | awk -F _ '{print $3}'`
setenv rundate `echo $j | awk -F _ '{print $2}'`

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

ls *.html >> ${WD}/LOG/log_${runnumber}

foreach i (`ls *.html`)
cat ${i} | sed s/LED/${CALIB}/g > ${i}_t
mv ${i}_t ${i} 
end

####### Copy to the new site
ls *.png
eos rm -rf ${PlotsDir}/${CALIB}_${runnumber}
eos mkdir ${PlotsDir}/${CALIB}_${runnumber} >> & ${WD}/LOG/logn_${runnumber}

if(${status} == "0") then
#### Copy to the new site
foreach i (`ls *.html`)
xrdcp ${i} ${PlotsDir}/${CALIB}_${runnumber} 
end
foreach k (`ls *.png`)
xrdcp ${k} ${PlotsDir}/${CALIB}_${runnumber}
end
endif



