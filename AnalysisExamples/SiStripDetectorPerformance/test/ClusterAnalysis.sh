#!/bin/sh

function usage(){
    echo -e "\n[usage] ClusterAnalysis.sh [options]"
    echo -e " -help  this message"
    echo -e " -InputFilePath=<path>"
    echo -e " -TestArea=<path>"
    echo -e " -Flag=<a flag>"
    echo -e " -CondDb=<sqlite>, <devdb10>, <orcon>, <orcoff> (default is orcoff)"
    echo -e " -sqliteDb=<dbfile> (needed for CondDb=sqlite - default is /tmp/$USER/dummy_<runNb>.db)"
    echo -e " -sqliteCatalog=<dbcatalog> (needed for CondDb=sqlite - default is /tmp/$USER/dummy_<runNb>.db )"
    echo -e " -castor (to get input files from castor)"
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
      inputfilenames="${inputfilenames},\"rfio:${InputFilePath}/$file\""
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

echo $@
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

cat template_ClusterAnalysis.cfg | sed -e "s#insert_DBfile#$DBfile#" -e "s#insert_DBcatalog#$DBcatalog#" -e "s#insert_root_filename#${root_filename}#" -e "s#insert_ps_filename#${ps_filename}#" -e "s#insert_input_file_list#$inputfilelist#" > ${cfg_file}
echo "cmsRun ${cfg_file}"
cmsRun ${cfg_file} > ${TestArea}/ClusterAnalysis_${Flag}.out


echo -e "\nroot file and postscript file with histos can be found in  ${TestArea}\n\t ${root_filename} \n\t ${ps_filename}" 
echo -e "\nto see .ps file do\ngv  ${ps_filename}&"
