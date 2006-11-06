#!/bin/sh
#Author domenico.giordano@cern.ch
 
function usage(){
    echo -e "\n[usage] ClusterAnalysis.sh [options]"
    echo -e " -help  this message"
    echo -e " -InputFilePath=<path>"
    echo -e " -TestArea=<path>"
    echo -e " -Flag=<a flag>"
    echo -e " -CondDb=<sqlite>, <devdb10>, <orcon>, <orcoff> (default is orcoff)"
    echo -e " -sqliteDb=<dbfile> (needed for CondDb=sqlite - default is /tmp/$USER/dummy_<runNb>.db)"
    echo -e " -sqliteCatalog=<dbcatalog> (needed for CondDb=sqlite - default is /tmp/$USER/dummy_<runNb>.db )"
    echo -e " -castor=<file name, or regular-expression> (to get input files from castor)"

    
    echo -e "\nEXAMPLES:"
    echo -e "\n\tSingle Local File access"
    echo -e "\n\t\t./ClusterAnalysis.sh -CondDb=orcoff -InputFilePath=/data/giordano/ClusterAnalysis/data/2501/reco_full_2501.root -Flag=Run2501" 

    echo -e "\n\tMultiple Local Files access"
    echo -e "\n\t\t./ClusterAnalysis.sh -CondDb=orcoff -InputFilePath=/data/giordano/ClusterAnalysis/data/25[0-4]\*/\*full\* -Flag=Runs2501-2549" 

    echo -e "\n\tSingle Castor File access"
    echo -e "\n\t\t./ClusterAnalysis.sh -CondDb=orcoff -InputFilePath=/castor/cern.ch/cms/testbeam/tkmtcc/P5_data/tracker_reprocessing/pass2 -castor=2501_reco_full.root -Flag=Run2501" 

    echo -e "\n\tMultiple Castor Files access (using regular expressions)"
    echo -e "\n\t\t./ClusterAnalysis.sh -CondDb=orcoff -InputFilePath=/castor/cern.ch/cms/testbeam/tkmtcc/P5_data/tracker_reprocessing/pass2 -castor='26\(\(3[7-9]\)\|\(4[0-2]\)\)_reco_full.root' -Flag=Runs2637-2642"  

    echo -e "\n\t\t./ClusterAnalysis.sh -CondDb=orcoff -InputFilePath=/castor/cern.ch/cms/testbeam/tkmtcc/P5_data/tracker_reprocessing/pass2_with_alignment -castor='26\(\(3[7-9]\)\|\(4[0-2]\)\)_reco_full.root' -Flag=Runs2637-2642_Align"  
    
    echo
    exit
}


function getLocalRunList(){      
#Create input file list
    
    inputfilenames=""
    for file in `ls ${InputFilePath}`
      do
      [ ! -e $file ] && continue
      inputfilenames="${inputfilenames},\"file:$file\""
    done
    
    inputfilenames=`echo $inputfilenames | sed -e "s@,@@"`
    echo $inputfilenames
}

function getCastorRunList(){      
#Create input file list
    inputfilenames=""
    for file in `nsls ${InputFilePath} | grep -E $castor 2> /dev/null`
      do
      inputfilenames="${inputfilenames},\"castor:${InputFilePath}/$file\""
    done
    
    inputfilenames=`echo $inputfilenames | sed -e "s@,@@"`
    echo $inputfilenames
}

function getRunList(){    

    if [ "$castor" != "0" ]; then
        getCastorRunList
    else
        getLocalRunList 
    fi
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

test_area=/tmp/$USER/ClusterAnalysis

getParameter InputFilePath $@ .
getParameter TestArea      $@ ${test_area}
getParameter Flag          $@ ""
getParameter CondDb        $@ orcoff
getParameter sqliteDb      $@ ${TestArea}/dummy_${run}.db
getParameter sqliteCatalog $@ ${TestArea}/dummy_${run}.xml
getParameter castor        $@ 0

[ ! -e ${TestArea} ] && mkdir -p ${TestArea}

if [ "$CondDb" == "sqlite" ] && [ "$sqliteDb" != "" ] && [ "$sqliteCatalog" != "" ]; 
    then
    DBfile="sqlite_file:${sqliteDb}"
    DBcatalog="file:${sqliteCatalog}"
elif [ "$CondDb" == "devdb10" ];  then
    DBfile="oracle://devdb10/CMS_COND_STRIP"
    DBcatalog="relationalcatalog_oracle://devdb10/CMS_COND_GENERAL"
elif [ "$CondDb" == "orcon" ]; then
    DBfile="oracle://orcon/CMS_COND_STRIP"
    DBcatalog="relationalcatalog_oracle://orcon/CMS_COND_GENERAL"
elif [ "$CondDb" == "orcoff" ]; then
    DBfile="oracle://cms_orcoff_int2r/CMS_COND_STRIP"
    DBcatalog="relationalcatalog_oracle://cms_orcoff_int2r/CMS_COND_GENERAL"
else
    echo -e "\nERROR: Wrong options"
    usage
fi

echo -e "\n -InputFilePath=$InputFilePath"
echo -e " -TestArea=$TestArea"
echo -e " -Flag=$Flag"
echo -e " -castor=$castor"
echo -e " -CondDb=$CondDb"
if [ "$CondDb" = "sqlite" ]; then
    echo -e " -sqliteDb=${sqliteDb}"
    echo -e " -sqliteCatalog=${sqliteCatalog}"
fi
echo " "

root_filename=${TestArea}/ClusterAnalysis_${Flag}.root 
ps_filename=${TestArea}/ClusterAnalysis_${Flag}.ps 
cfg_file=${TestArea}/ClusterAnalysis_${Flag}.cfg

echo ${cfg_file}

export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb

eval `scramv1 runtime -sh`

inputfilelist=`getRunList`

[ "$inputfilelist" == "" ] && echo "No file exists for the specified path" && exit

cat template_ClusterAnalysis.cfg | sed -e "s#insert_DBfile#$DBfile#" -e "s#insert_DBcatalog#$DBcatalog#" -e "s#insert_root_filename#${root_filename}#" -e "s#insert_ps_filename#${ps_filename}#" -e "s#insert_input_file_list#$inputfilelist#" > ${cfg_file}
echo "cmsRun ${cfg_file}"
cmsRun ${cfg_file} > ${TestArea}/ClusterAnalysis_${Flag}.out

echo -e "\nlog file " ${TestArea}/ClusterAnalysis_${Flag}.out
echo
echo -e "\nroot file and postscript file with histos can be found in  ${TestArea}\n\t ${root_filename} \n\t ${ps_filename}" 
echo -e "\nto see .ps file do\ngv  ${ps_filename}&"
