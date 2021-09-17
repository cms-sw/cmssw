#!/bin/csh

set runnumber=${1}
set refnumber=${3}
set runNevents=${4}
set CALIB=${5}

set RELEASE=CMSSW_10_4_0

set fullSrc0='/store/group/dpg_hcal/comm_hcal/USC'
set fullSrc1='/store/group/dpg_hcal/comm_hcal/LS1'
set fullSrc='NO'
set WebDir='/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMweb'
set WebSite='https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT'
set HistoDir='/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMweb/histos'
set WD="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/RecoHcal/HcalPromptAnalysis/test/RDM"

### We are at working node
mkdir ${runnumber}
setenv WORK `pwd`/${runnumber}

echo ${WORK} >> ${WD}/LOG/batchlog

cd /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/${RELEASE}/src/RecoHcal/HcalPromptAnalysis/test
cmsenv
###cp ${WD}/remoteMonitoring_${CALIB}_cfg.py ${WORK}/remoteMonitoring_cfg.py

ls ${WD}/RemoteMonitoringMAP.cc.exe  >> ${WD}/LOG/batchlog

cp ${WD}/RemoteMonitoringMAP.cc ${WORK}
cp ${WD}/compile.csh ${WORK}
cp ${WD}/LogEleMapdb.h ${WORK}
cp ${WD}/${CALIB}_LIST/runlist.tmp.${2} ${WORK}/runlist.tmp

cd ${WORK}

./compile.csh RemoteMonitoringMAP.cc

#### cmsRun Start
### Temporarily
#rm LOG/log_${runnumber}
#rm ${HistoDir}/${CALIB}_${runnumber}.root

echo " Start CMS run ">${WD}/LOG/logn_${runnumber}
echo ${LD_LIBRARY_PATH} >>${WD}/LOG/logn_${runnumber}

#cmsRun remoteMonitoring_cfg.py ${runnumber} ${fullSrc} ${HistoDir} >> & ${WD}/LOG/log_${runnumber}
#mv ${HistoDir}/LED_${runnumber}.root ${HistoDir}/${CALIB}_${runnumber}.root

rm -rf ${WebDir}/${CALIB}_${runnumber} 
mkdir ${WebDir}/${CALIB}_${runnumber} >> & ${WD}/LOG/logn_${runnumber}
#####./compile.csh RemoteMonitoringMAP.cc  >> & ${WD}/LOG/logn_${runnumber}
./RemoteMonitoringMAP.cc.exe "${HistoDir}/${CALIB}_${runnumber}.root" "${HistoDir}/${CALIB}_${refnumber}.root" "${CALIB}" >> ${WD}/LOG/logn_${runnumber}

##root -b -q -l 'RemoteMonitoringMAP.C+("'${HistoDir}'/${CALIB}_'${runnumber}'.root","'${HistoDir}'/${CALIB}_'${refnumber}'.root")'
ls -l >> ${WD}/LOG/log_${runnumber}
#echo " Start copy png " >> & ${WD}/LOG/log_${runnumber}
#ls $WebDir/${CALIB}_$runnumber  >> & ${WD}/LOG/log_${runnumber}
#cp *.html $WebDir/${CALIB}_$runnumber  >> & ${WD}/LOG/log_${runnumber}
#cp *.png $WebDir/${CALIB}_$runnumber  >> & ${WD}/LOG/log_${runnumber}

#### CmsRun end
### extract the date of file
### set rundate=`cmsLs $fullSrc | grep ${1} | awk '{print $3}'`

set j=`cat runlist.tmp | grep ${runnumber}`
echo ${j} >> ${WD}/LOG/batchlog    
setenv runtype ${CALIB}
setenv runHTML NO
setenv runtime `echo $j | awk -F _ '{print $4}'`
setenv rundate `echo $j | awk -F _ '{print $3}'`
#wget ${runHTML} >> ${WD}/LOG/batchlog
#setenv runNevents `cat index.html | tail -n +14 | head -n 1 | awk -F '>' '{print $2}' | awk -F '<' '{print $1}'`
#rm index.html

echo 'RUN Date = '${rundate} ${runtime} >> ${WD}/LOG/batchlog    
echo 'RUN Type = '${runtype} >> ${WD}/LOG/batchlog    
echo 'Reference RUN number ='${refnumber} >> ${WD}/LOG/batchlog    

touch index_draft.html

#adding entry to list of file index_draft.html

####sed '$d' < /afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMweb/index.html > index.html.tmp

set raw=3
echo '<tr>'>> index_draft.html
echo '<td class="s1" align="center">'ktemp'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$runnumber'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$runtype'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$runNevents'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$rundate'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$runtime'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$refnumber'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center"><a href="'$WebSite'/'${CALIB}'_'$runnumber'/MAP.html">'${CALIB}'_'$runnumber'</a></td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">NO</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">OK</td>'>> index_draft.html
echo '</tr>'>> index_draft.html

#### PUT Corresponding calib type to html

ls *.html >> ${WD}/LOG/log_${runnumber}

foreach i (`ls *.html`)
cat ${i} | sed s/LED/${CALIB}/g > ${i}_t
mv ${i}_t ${i} 
end

#######touch $WebDir/${CALIB}_$runnumber/new
####### Copy to the new site in parallel
ls *.png
if(${status} == "0") then
#### Copy to the old site
echo " Start copy png " >> & ${WD}/LOG/log_${runnumber}
cp *.html $WebDir/${CALIB}_$runnumber  >> & ${WD}/LOG/log_${runnumber}
cp *.png $WebDir/${CALIB}_$runnumber  >> & ${WD}/LOG/log_${runnumber}

mv $WebDir/${CALIB}_$runnumber/index_draft.html $WebDir/${CALIB}_$runnumber/index_draft.html.orig
cp index_draft.html $WebDir/${CALIB}_$runnumber
#### Copy to the new site
eos mkdir /eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT/${CALIB}_$runnumber
foreach i (`ls *.html`)
cat ${i} | sed 's#cms-cpt-software.web.cern.ch\/cms-cpt-software\/General\/Validation\/SVSuite#cms-conddb-dev.cern.ch\/eosweb\/hcal#g'> ${i}.n
xrdcp ${i}.n /eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT/${CALIB}_$runnumber
end
foreach k (`ls *.png`)
xrdcp ${k} /eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/RMT/${CALIB}_$runnumber
end
endif



