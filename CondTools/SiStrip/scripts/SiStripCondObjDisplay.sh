#!/bin/sh

function usage(){
    echo -e "\n[usage] SiStripCondObjDisplay.sh [options]"
    echo -e " -help  this message"
    echo -e " -run=<runNb>"
    echo -e " -CondDb=<sqlite>, <devdb10>, <orcon> (default is sqlite)"
    echo -e " -dbfile=<dbfile> (needed for CondDb=sqlite - default is /tmp/$USER/dummy_<runNb>.db)"
    echo -e " -dbcatalog=<dbcatalog> (needed for CondDb=sqlite - default is /tmp/$USER/dummy_<runNb>.db )"
    exit
}

function getParameter(){
    what=$1
    eval $what=\$$#
    shift
    where=$@
    if [ `echo $where | grep -c "\-$what="` = 1 ]; then
        eval $what=`echo $where | awk -F"${what}=" '{print $2}' | awk '{print $1}'`
    elif [ `echo $where | grep -c "\-$what"` = 1 ]; then
	eval $what=1
    fi
}

#################
## MAIN
#################

[ `echo $@ | grep -c "\-help"` = 1 ] && usage;

test_area=/tmp/$USER/Display
[ ! -e ${test_area} ] && mkdir -p ${test_area}

getParameter run       $@ -1
getParameter CondDb    $@ sqlite
getParameter dbfile    $@ ${test_area}/dummy_${run}.db
getParameter dbcatalog $@ ${test_area}/dummy_${run}.xml

[ "$run" == "-1" ] && echo -e "\nWORNING: please specify a run number" && usage

if [ "$CondDb" == "sqlite" ] && [ "$dbfile" != "" ] && [ "$dbcatalog" != "" ]; 
    then
    DBfile="sqlite_file:${dbfile}"
    DBcatalog="file:${dbcatalog}"
elif [ "$CondDb" == "devdb10" ];  then
    DBfile="oracle://devdb10/CMS_COND_STRIP"
    DBcatalog="relationalcatalog_oracle://devdb10/CMS_COND_GENERAL"
elif [ "$CondDb" == "orcon" ]; then
    DBfile="oracle://orcon/CMS_COND_STRIP"
    DBcatalog="relationalcatalog_oracle://orcon/CMS_COND_GENERAL"
else
    echo -e "\nERROR: Wrong options"
    usage
fi



output_file_name=${test_area}/Display_PedNoise_RunNb_${run}.root 
ps_file_name=${test_area}/Display_PedNoise_RunNb_${run}.ps 
cfg_file=${test_area}/SiStripCondObjDisplay_RunNb_${run}.cfg

echo ${cfg_file}

export TNS_ADMIN=/afs/cern.ch/project/oracle/admin
export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb

eval `scramv1 runtime -sh`

cat template_SiStripCondObjDisplay.cfg | sed -e "s#insert_DBfile#$DBfile#" -e "s#insert_DBcatalog#$DBcatalog#" -e "s#insert_output_filename#${output_file_name}#" -e "s#insert_ps_filename#${ps_file_name}#" -e "s#insert_runNb#${run}#" > ${cfg_file}
echo "cmsRun ${cfg_file}"
cmsRun ${cfg_file} > ${test_area}/out_diplay_${run}

grep "SiStripCondObjDisplay::geometry_Vs_Cabling"  ${test_area}/out_diplay_${run}
echo
echo
echo "Empty Pedestal Histos" `grep "Empty Ped"  ${test_area}/out_diplay_${run} | wc -l`
echo "Empty Noise Histos" `grep "Empty Nois"  ${test_area}/out_diplay_${run} | wc -l`
echo "Empty BadStrip Histos" `grep "Empty Bad"  ${test_area}/out_diplay_${run} | wc -l`
echo "Pedestal Histos with all channel at zero" `grep "All channel at zero in Ped"  ${test_area}/out_diplay_${run} | wc -l`
echo "Noise Histos with all channel at zero" `grep "All channel at zero in Nois"  ${test_area}/out_diplay_${run} | wc -l`
echo "BadStrips Histos with all channel at zero" `grep "All channel at zero in Bad"  ${test_area}/out_diplay_${run} | wc -l`

echo -e "\nroot file and postscript file with histos can be found in  ${test_area}\n\t ${output_file_name} \n\t ${ps_file_name}" 
echo -e "\nto see .ps file do\ngv  ${ps_file_name}&"
