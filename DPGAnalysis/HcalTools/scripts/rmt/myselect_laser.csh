#!/bin/csh
###
### Day/Time
### All files have the particular time
###
#set DAT=`date '+%Y-%m-%d_%H_%M_%S'`
set WD='/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/CMSSW_5_3_21/src/RecoHcal/HcalPromptAnalysis/test/RDM'
set REF=224708
set LAST=`cat ${WD}/LASER_LIST/LASTRUN` 
if( ${LAST} == "" ) then
echo  " Network problem: no access to the LASTRUN"
exit
endif

#### Uncomment after it is done

#wget -q http://test-dtlisov.web.cern.ch/test-dtlisov/ -O ${WD}/LASER_LIST/done.html.${1}
#touch ${WD}/LASER_LIST/done.html.${1}
#touch ${WD}/LASER_LIST/goodledruns.${1}

wget -q http://cmshcalweb01.cern.ch/DetDiag/Local_HTML/runlist.html -O ${WD}/LASER_LIST/runlist.html.${1}

cat ${WD}/LASER_LIST/runlist.html.${1} | sed s/\"//g | sed st\>\/t' 'tg | sed sk\<\/td\>kkg | sed sk\<\/a\>kkg | tr '\n' ' ' | awk -F '</tr>' '{for(i=1;i<=NF;i++) printf $i"\n"}' | awk -F '<tr> <td' '{print $2}' | tail -n +4 | sed s/' '/-/g | grep LASER > ${WD}/LASER_LIST/runlist.tmp0.${1}

touch ${WD}/LASER_LIST/runlist.tmp.${1}

set count=0

foreach i (`cat ${WD}/LASER_LIST/runlist.tmp0.${1}`)
#echo ${i}
set TYPE=`echo $i | awk -F - '{print $13}'`
set HTML=`echo $i | awk -F - '{print $25}' | awk -F 'href=' '{print $2}'`
set RUNNUMBER=`echo $i | awk -F - '{print $9}'`
set YEAR=`echo $i | awk -F - '{print $17}'`
set MONTH=`echo $i | awk -F - '{print $18}'`
set DAY=`echo $i | awk -F - '{print $19}'`
set TIME=`echo $i | awk -F - '{print $20}'`

echo "Type" ${TYPE} 
echo " RunNumber" ${RUNNUMBER}

if ( ${TYPE} == "LASER" && ${RUNNUMBER} > ${LAST} ) then
echo ${TYPE} ${HTML} ${RUNNUMBER}
wget -q ${HTML} -O ${WD}/LASER_LIST/index.html.${1}
set NEVENTS=`cat ${WD}/LASER_LIST/index.html.${1} | tail -n +14 | head -n 1 | awk -F '>' '{print $2}' | awk -F '<' '{print $1}'`
echo ${RUNNUMBER} "Number of events" ${NEVENTS}
#if( ${NEVENTS} >= "2000" && ${NEVENTS} < "10000" && ${RUNNUMBER} > ${LAST} ) then
#if( ${NEVENTS} >= "10000" && ${RUNNUMBER} > ${LAST} ) then
echo ${RUNNUMBER} "Number of events" ${NEVENTS} >> ${WD}/LASER_LIST/goodledruns.${1}
#grep -q ${RUNNUMBER} ${WD}/LASER_LIST/done.html.${1}
#if( ${status} == "1") then
echo "${RUNNUMBER}_${REF}_${YEAR}-${MONTH}-${DAY}_${TIME}_${NEVENTS}" >> ${WD}/LASER_LIST/runlist.tmp.${1} 
if( ${count} == "0" ) then
rm LASER_LIST/LASTRUN
echo ${RUNNUMBER} > LASER_LIST/LASTRUN
endif
@ count = ${count} + "1"
#endif
#endif
rm ${WD}/LASER_LIST/index.html.${1}
endif
end

#rm ${WD}/LASER_LIST/done.html.${1}
#rm ${WD}/LASER_LIST/runlist.html.${1}

cmsLs /store/group/dpg_hcal/comm_hcal/USC > ${WD}/LASER_LIST/fullSrc0_list_${1}
cmsLs /store/group/dpg_hcal/comm_hcal/LS1 > ${WD}/LASER_LIST/fullSrc1_list_${1}




