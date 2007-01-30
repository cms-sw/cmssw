  #!/bin/sh
    
function usage () {
    echo -e "\n[usage]\n SiStripOfflinePedNoiseToDb.sh [options]"
    echo -e " -InputFilePath=<path>"
    echo -e " -castor=<file name, or regular-expression> (to get input files from castor)"
    echo -e " -TestArea=<path>"
    echo -e " -StartIOV=<runNb> (default is ${default_StartIOV} )"
    echo -e " -tagPN=<tag for PedNoise> (default is ${default_tagPN} )"
    echo -e " -tagCab=<tag for cabling> (default is ${default_tagCab} )"
    echo -e " -doPedNoiseTransfer (default is $default_doPedNoiseTransfer )"
    echo -e " -CondDb=<sqlite>, <devdb10>, <orcon> (default is sqlite)"
    echo -e " -sqliteDb=<path_name> (default is /tmp/$USER/OffPedNoiDb/dummy_<StartIOV>.db)"
    echo -e " -sqliteCatalog=<path_name> (default is /tmp/$USER/OffPedNoiDb/dummy_<StartIOV>.xml)"
    echo -e " -firstUpload (otherwise works in append mode) "
    echo -e " -geometry=<TAC>, <MTCC> (default is MTCC)"


    echo -e "\n\nPlease set your CORAL_AUTH_PATH environment variable, otherwise it will be defined as /afs/cern.ch/cms/DB/conddb\n"

    echo -e "\nEXAMPLES:"
    echo -e "\n\tSingle Local File access"
    echo -e "\n\t\t./SiStripOfflinePedNoiseToDb.sh -tagPN=SiStripOffPedNoi_v1 -tagCab=SiStripCabling_v1 -doPedNoiseTransfer -CondDb=devdb10 -geometry=TAC -InputFilePath=/storage/TIB/run/RU0000518_000.root  -StartIOV=100 -firstUpload"

    echo -e "\n\t\t ./SiStripOfflinePedNoiseToDb.sh -tagPN=SiStripOffPedNoi_v1 -tagCab=SiStripCabling_v1 -doPedNoiseTransfer -geometry=TAC -InputFilePath=/storage/TIB/run/RU0000516_000.root -StartIOV=516 -sqliteDb=/tmp/giordano/o2o/dummy_1.db -sqliteCatalog=/tmp/giordano/o2o/dummy_1.xml -firstUpload"

    echo -e "\n\t\t ./SiStripOfflinePedNoiseToDb.sh -tagPN=SiStripOffPedNoi_v1 -tagCab=SiStripCabling_v1 -doPedNoiseTransfer -geometry=TAC -InputFilePath=/storage/TIB/run/RU0000516_000.root -StartIOV=600 -sqliteDb=/tmp/giordano/o2o/dummy_1.db -sqliteCatalog=/tmp/giordano/o2o/dummy_1.xml "


    echo -e "\n\tMultiple Local Files access"
    echo -e "\n\t\t./SiStripOfflinePedNoiseToDb.sh -tagPN=SiStripOffPedNoi_v1 -tagCab=SiStripCabling_v1 -doPedNoiseTransfer -CondDb=devdb10 -geometry=TAC -InputFilePath=/storage/TIB/run/RU000052[0-4]\*  -StartIOV=100" 

    echo -e "\n\tSingle Castor File access"
    echo -e "\n\t\t./SiStripOfflinePedNoiseToDb.sh -tagPN=SiStripOffPedNoi_v1 -tagCab=SiStripCabling_v1 -doPedNoiseTransfer -CondDb=devdb10 -geometry=TAC -InputFilePath=/castor/cern.ch/cms/testbeam/tkmtcc/P5_data/tracker_reprocessing/pass2 -castor=2501_reco_full.root -StartIOV=100" 

    echo
    exit
}

function getParameter(){
    what=$1
    shift
    where=$@
    if [ `echo $where | grep -c "\-$what="` = 1 ]; then
        eval $what=`echo $where | awk -F"${what}=" '{print $2}' | awk '{print $1}'`
    elif [ `echo $where | grep -c "\-$what "` = 1 ]; then
	eval $what=1
    else
	let c=$#-1
	shift $c
	eval $what=$1
    fi
}

function settings (){

    default_TestArea=/tmp/$USER/OfflinePedNoiseToDb
    default_StartIOV=1
    default_tagPN=SiStripPedNoise_v1
    default_tagCab=SiStripCabling_v1
    default_doPedNoiseTransfer=0
    default_CondDb="sqlite"
    default_firstUpload=0
    
    getParameter help $@ 0
    [ "$help" = 1 ] && usage

    if [ ! -n "$CORAL_AUTH_PATH" ];
	then
	export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb
	echo -e "\nWARNING: CORAL_AUTH_PATH environment variable is not defined in your shell\n default value will be used CORAL_AUTH_PATH=$CORAL_AUTH_PATH"
    fi

    getParameter InputFilePath        $@ .
    getParameter TestArea             $@ ${default_TestArea}
    getParameter StartIOV             $@ ${default_StartIOV}
    getParameter tagPN                $@ ${default_tagPN}
    getParameter tagCab               $@ ${default_tagCab}
    getParameter doPedNoiseTransfer   $@ ${default_doPedNoiseTransfer}
    getParameter CondDb               $@ ${default_CondDb}
    getParameter firstUpload          $@ ${default_firstUpload} 
    getParameter geometry             $@ MTCC
    getParameter castor               $@ 0

    default_sqliteDb=${TestArea}/dummy_${StartIOV}.db
    default_sqliteCatalog=${TestArea}/dummy_${StartIOV}.xml
    getParameter sqliteDb             $@ ${default_sqliteDb}
    getParameter sqliteCatalog        $@ ${default_sqliteCatalog}


    [ ! -e ${sqliteDb} ] && [ "$CondDb" == "sqlite" ] && firstUpload=1

    append=1
    [ "$firstUpload" = 1 ] && append=0
    export append
    echo -e "\n -InputFilePath=$InputFilePath"
    echo -e " -TestArea=$TestArea"
    echo -e " -castor=$castor"    
    echo -e " -StartIOV=$StartIOV"
    echo -e " -CondDb=$CondDb"
    echo -e " -tagPN=$tagPN"
    echo -e " -tagCab=$tagCab"
    echo -e " -firstUpload $firstUpload"
    echo -e " -doPedNoiseTransfer $doPedNoiseTransfer"
    echo -e " -geometry=$geometry"
    if [ "$CondDb" = "sqlite" ]; then
	echo -e " -sqliteDb=${sqliteDb}"
	echo -e " -sqliteCatalog=${sqliteCatalog}"
    fi
    echo " "
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
    shift
    where=$@
    if [ `echo $where | grep -c "\-$what="` = 1 ]; then
        eval $what=`echo $where | awk -F"${what}=" '{print $2}' | awk '{print $1}'`
    elif [ `echo $where | grep -c "\-$what"` = 1 ]; then
	eval $what=1
    else
	let c=$#-1
	shift $c
	eval $what=$1
    fi
}

#################
## MAIN
#################


settings "$@"


[ ! -e ${TestArea} ] && mkdir -p ${TestArea}


eval `scramv1 runtime -sh`
export TNS_ADMIN=/afs/cern.ch/project/oracle/admin
  
if [ "$CondDb" == "devdb10" ]; then
    DBfile="oracle://devdb10/CMS_COND_STRIP"
    DBcatalog="relationalcatalog_oracle://devdb10/CMS_COND_GENERAL"
elif [ "$CondDb" == "orcon" ]; then
    DBfile="oracle://orcon/CMS_COND_STRIP"
    DBcatalog="relationalcatalog_oracle://orcon/CMS_COND_GENERAL"
elif [ "$CondDb" == "sqlite" ]; then
    DBfile="sqlite_file:${sqliteDb}"
    DBcatalog="file:${sqliteCatalog}"

#     if [ "${append}" == 0 ] ;
# 	then 
# 	rm -f ${sqliteDb}
# 	rm -f ${sqliteCatalog}
        
# 	echo "cmscond_bootstrap_detector.pl --offline_connect $DBfile --auth ${CORAL_AUTH_PATH}/authentication.xml STRIP "
# 	cmscond_bootstrap_detector.pl --offline_connect $DBfile --auth ${CORAL_AUTH_PATH}/authentication.xml STRIP 
# 	pool_insertFileToCatalog -u ${DBcatalog} -t POOL_RDBMS ${DBfile}
# 	echo " " 
#     fi
else
    echo "ERROR: wrong argument value: -CondDb=<sqlite>, <devdb10>, <orcon> "
    echo "EXIT"
    exit
fi


o2otrans="dummy"
[ "$doPedNoiseTransfer" == "1" ]    && o2otrans="#o2o_pednoi"


cfg_file=${TestArea}/SiStripOffPedNoisDb_StartIOV_${StartIOV}.cfg

echo DBfile $DBfile
echo DBcatalog $DBcatalog

inputfilelist=`getRunList`
[ "$inputfilelist" == "" ] && echo "No file exists for the specified path" && exit

templatefile=${CMSSW_BASE}/src/CondTools/SiStrip/scripts/template_SiStripOfflinePedNoiseToDb.cfg 
[ ! -e $templatefile ] && templatefile=${CMSSW_RELEASE_BASE}/src/CondTools/SiStrip/scripts/template_SiStripOfflinePedNoiseToDb.cfg 
[ ! -e $templatefile ] && echo "ERROR: expected template file doesn't exist both in your working area and in release area. Please fix it." && exit

cat $templatefile | sed  -e "s@#${geometry}@@g" -e "s#insert_DBfile#$DBfile#g" -e "s#insert_DBcatalog#$DBcatalog#g"  -e "s#insert_StartIOV#${StartIOV}#" -e "s#insert_appendflag#${append}#g" -e "s@#appendMode_${append}@@g" -e "s#insert_input_file_list#$inputfilelist#" -e "s#insert_tagPN#${tagPN}#g"  -e "s#insert_tagCab#${tagCab}#g" -e "s#insert_doPedNoiseTransfer#$doPedNoiseTransfer#" -e "s@${o2otrans}@@"> ${cfg_file}

echo -e "\ncmsRun ${cfg_file}"
cmsRun ${cfg_file} > ${TestArea}/out_${StartIOV}

echo -e "\nlog file " ${TestArea}/out_${StartIOV}.out

