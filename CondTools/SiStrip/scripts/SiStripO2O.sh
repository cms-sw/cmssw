#!/bin/sh

function usage () {
    echo -e "\n[usage]\n SiStripO2O.sh [options]"
    echo -e " -Run=<runNb> (default is ${default_Run} )"
    echo -e " -tag=<tag> (default is ${default_tag} it will create SiStripPedNoise_<tag>_p, SiStripPedNoise_<tag>_n, SiStripCabling_<tag>, )"
    echo -e " -ConfigDb=<user/passwd@path> (default is ${default_ConfigDb} )"
    echo -e " -ConfigDbVersion=<Major.Minor> (default is ${default_ConfigDb} )"
    echo -e " -ConfigDbPartition=<partitionName> (default is ${default_ConfigDbPartition} )"
    echo -e " -doPedNoiseTransfer (default is $default_doPedNoiseTransfer )"
    echo -e " -doFedCablingTransfer (default is $default_doFedCablingTransfer )" 
    echo -e " -CondDb=<sqlite>, <devdb10>, <orcon> (default is sqlite)"
    echo -e " -sqliteDb=<path_name> (default is /tmp/$USER/o2o/dummy_<Run>.db)"
    echo -e " -sqliteCatalog=<path_name> (default is /tmp/$USER/o2o/dummy_<Run>.xml)"
    echo -e " -firstUpload (otherwise works in append mode) "
    echo -e " -geometry=<TAC>, <MTCC> (default is TAC)"
    echo -e " -Debug (switch on printout for debug)"
    echo -e " -RunTable (switch on/off online runtable query - default off)"
    echo -e " -force (Force the o2o for given run)"
    echo -e "\n\nPlease set your CORAL_AUTH_PATH environment variable, otherwise it will be defined as /afs/cern.ch/cms/DB/conddb\n"
    
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

    export test_area=/tmp/$USER/o2o

    default_Run=1
    default_tag="v1"
    default_doPedNoiseTransfer=1
    default_doFedCablingTransfer=1
    default_ConfigDb=cms_tracker_tif2/stabletif2@cms_tec_lyon
    default_ConfigDbVersion=1.0
    default_ConfigDbPartition="TIBTOB_TEST1"
    default_CondDb="sqlite"
    default_firstUpload=0
    default_Debug=0
    default_RunTable=1
    default_geometry=TAC
    default_force=0

    getParameter help $@ 0
    [ "$help" = 1 ] && usage

    if [ ! -n "$CORAL_AUTH_PATH" ];
	then
	export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb
#	export CORAL_AUTH_PATH=.auth
	#echo -e "\nWARNING: CORAL_AUTH_PATH environment variable is not defined in your shell\n default value will be used CORAL_AUTH_PATH=$CORAL_AUTH_PATH\n"
    fi
    echo ""
    
    getParameter doPedNoiseTransfer   $@ ${default_doPedNoiseTransfer}
    getParameter doFedCablingTransfer $@ ${default_doFedCablingTransfer}
    getParameter ConfigDb             $@ ${default_ConfigDb}
    getParameter ConfigDbVersion      $@ ${default_ConfigDbVersion}
    getParameter ConfigDbPartition    $@ ${default_ConfigDbPartition}
    getParameter Run                  $@ ${default_Run}
    getParameter tag                  $@ ${default_tag}
    getParameter CondDb               $@ ${default_CondDb}
    getParameter firstUpload          $@ ${default_firstUpload} 
    getParameter geometry             $@ ${default_geometry}
    getParameter Debug                $@ ${default_Debug}
    getParameter RunTable             $@ ${default_RunTable}
    getParameter RunTable             $@ ${default_RunTable}
    getParameter force                $@ ${default_force}

    default_sqliteDb=${test_area}/dummy_${Run}.db
    default_sqliteCatalog=${test_area}/dummy_${Run}.xml
    getParameter sqliteDb             $@ ${default_sqliteDb}
    getParameter sqliteCatalog        $@ ${default_sqliteCatalog}

    [ ! -e ${sqliteDb} ] && [ "$CondDb" == "sqlite" ] && firstUpload=1

    append=1
    [ "$firstUpload" = 1 ] && append=0
    export append
    
    #//Config DB parameters
    ConfigDbUser=`echo ${ConfigDb}| awk -F'/' '{print $1}'`
    ConfigDbPasswd=`echo ${ConfigDb}| awk -F'/' '{print $2}' | awk -F'@' '{print $1}'`
    ConfigDbPath=`echo ${ConfigDb}| awk -F'@' '{print $2}'`
    ConfigDbMajorVersion=`echo ${ConfigDbVersion}| awk -F'.' '{print $1}'`
    ConfigDbMinorVersion=`echo ${ConfigDbVersion}| awk -F'.' '{print $2}'`

    [ "${RunTable}" == "1" ] && QueryOnline

    #//Set CondDB tags
    
    if [ `echo $ConfigDbPartition | grep -c -i TIBTOB` == '1' ]; then
	tag=TIBTOB_${tag}
    elif [ `echo $ConfigDbPartition | grep -c -i TIB`    == '1' ]; then
	tag=TIB_${tag}
    elif [ `echo $ConfigDbPartition | grep -c -i TOB`    == '1' ]; then
	tag=TOB_${tag}
    elif [ `echo $ConfigDbPartition | grep -c -i TEC`    == '1' ]; then
	tag=TEC_${tag}
    fi

    export tagCab=SiStripCabling_${tag}
    export tagPN=SiStripPedNoise_${tag}

    echo -e " -Run=$Run"
    echo -e " -ConfigDb=${ConfigDb}"
    echo -e " -ConfigDbVersion=${ConfigDbVersion}"
    echo -e " -ConfigDbPartition=${ConfigDbPartition}"
    echo -e " -CondDb=$CondDb"
    echo -e " -tag=$tag"
    echo -e " -geometry=$geometry"
    echo -e " -firstUpload $firstUpload"
    echo -e " -doPedNoiseTransfer $doPedNoiseTransfer"
    echo -e " -doFedCablingTransfer $doFedCablingTransfer"
    if [ "$CondDb" = "sqlite" ]; then
	echo -e " -sqliteDb=${sqliteDb}"
	echo -e " -sqliteCatalog=${sqliteCatalog}"
    fi
    echo -e " -Debug $Debug"
    echo -e " -force $force"

    echo " "
}


function QueryOnline(){

    Where="partition,run,state,modetype"
    Condition="run.runmode=modetype.runmode and state.stateid=run.stateid and state.partitionid=run.partitionid and partition.partitionid=run.partitionid and run.runnumber=$Run"

    answer=`echo "select run.runmode||' '||modetype.modedescription||' '||partitionname||' '||state.FEDVERSIONMAJORID||' '||state.FEDVERSIONMINORID from ${Where} where ${Condition};" | sqlplus -S ${ConfigDb} | tail -2 | head -1 `

    export ConfigDbRunMode=`echo $answer | awk '{print $1}'`
    export ConfigDbRunDesc=`echo $answer | awk '{print $2}'`
    export ConfigDbPartition=`echo $answer | awk '{print $3}'`
    export ConfigDbMajorVersion=`echo $answer | awk '{print $4}'`
    export ConfigDbMinorVersion=`echo $answer | awk '{print $5}'`

    if  [ "${ConfigDbRunMode}" != "1" ] && [ "$force" == "0" ] ;
	then
	echo -e "\n RunMode from RunTable on $ConfigDb is ${ConfigDbRunMode} = ${ConfigDbRunDesc}"
	echo -e "\n RunMode doesn't match allowed modes"
	echo -e "\n\t  1 = Physics"
	echo -e "\n EXIT"
	exit
    fi

    ConfigDbVersion=${ConfigDbMajorVersion}.${ConfigDbMinorVersion}

}

function VerifyAppendMode(){ 
    
    connectionName=$1

    account=`grep -h -A2 "$1" ${CORAL_AUTH_PATH}/*.xml | sed -e 's@<[[:print:]]*value="\([[:print:]]*\)"[[:print:]]*@\1@g' | tail -2 | head -1 | sed -e "s@[ \t]@@g"`
    passwd=`grep -h -A2 "$1" ${CORAL_AUTH_PATH}/*.xml | sed -e 's@<[[:print:]]*value="\([[:print:]]*\)"[[:print:]]*@\1@g' | tail -1 | sed -e "s@[ \t]@@g"`

 #   echo $account $passwd

    declare -i value
    [ "$doFedCablingTransfer" == "1" ] && value=`echo "select count(NAME) from  CMS_COND_STRIP.metadata where name like '${tagCab}';"  | sqlplus -S ${account}/${passwd}@${CondDb} | tail -2 | head -1 | sed -e "s@[ \t]@@g"`
    [ "$doPedNoiseTransfer" == "1" ]   && value=`echo "select count(NAME) from  CMS_COND_STRIP.metadata where name like '${tagPN}_p';" | sqlplus -S ${account}/${passwd}@${CondDb} | tail -2 | head -1 | sed -e "s@[ \t]@@g"`

    append=1
    [ "$value" == "0" ] && append=0
}


function VerifyIfO2ONeeded(){
    condition=`echo -e "$ConfigDb \t $ConfigDbPartition \t ${ConfigDbMajorVersion}.${ConfigDbMinorVersion} \t $CondDb \t $tag"`
    if [ `tail -1 .SiStripO2OTable.log | grep -c "$condition"`  != "0" ]; 
	then
	echo -e "\n The following condition is already uploaded. O2O not needed"
	echo -e "\n $condition"
	echo -e "\n EXIT"
	exit	
    fi

    if [ `grep ${CondDb} .SiStripO2OTable.log | grep -c ${tag}` != "0" ] && [ `grep ${tag} .SiStripO2OTable.log | grep ${CondDb} | tail -1 | awk '{print $6}'`  -ge $Run ]; 
	then
	echo -e "\n You are trying to upload condition for the run $Run that comes before the last IOV uploaded"
	tail -1 .SiStripO2OTable.log
	echo -e "\n EXIT"
	exit	
    fi
}

#################
## MAIN
#################

#@@@@@@@@@@@@@@@
#TEMP
#cd /exports/xdaq/CMSSW/Development/Domenico/CurrentO2O/CMSSW_1_2_0/src/
#@@@@@@@@@@@@@@@

eval `scramv1 runtime -sh`
export TNS_ADMIN=/afs/cern.ch/project/oracle/admin

#@@@@@@@@@@@@@@@
#TEMP
#cd -
#@@@@@@@@@@@@@@@
  
[ ! -e  .SiStripO2OTable.log ] && touch  .SiStripO2OTable.log

settings "$@"

[ ! -e ${test_area} ] && mkdir -p ${test_area}


if [ "$CondDb" == "devdb10" ]; then
    DBfile="oracle://devdb10/CMS_COND_STRIP"
    DBcatalog="relationalcatalog_oracle://devdb10/CMS_COND_GENERAL"
    VerifyAppendMode $DBfile
elif [ "$CondDb" == "orcon" ]; then
    DBfile="oracle://orcon/CMS_COND_STRIP"
    DBcatalog="relationalcatalog_oracle://orcon/CMS_COND_GENERAL"
    VerifyAppendMode $DBfile
elif [ "$CondDb" == "sqlite" ]; then
    DBfile="sqlite_file:${sqliteDb}"
    DBcatalog="file:${sqliteCatalog}"

    if [ "${append}" == 0 ] ;
	then 
	rm -f ${sqliteDb}
	rm -f ${sqliteCatalog}
        
	echo "cmscond_bootstrap_detector.pl --offline_connect $DBfile --auth ${CORAL_AUTH_PATH}/authentication.xml STRIP "
	cmscond_bootstrap_detector.pl --offline_connect $DBfile --auth ${CORAL_AUTH_PATH}/authentication.xml STRIP 
	pool_insertFileToCatalog -u ${DBcatalog} -t POOL_RDBMS ${DBfile}
	echo " " 
    fi
else
    echo "ERROR: wrong argument value: -CondDb=<sqlite>, <devdb10>, <orcon> "
    echo "EXIT"
    exit
fi

o2otrans="dummy"
if [ "$doPedNoiseTransfer" == "1" ] && [ "$doFedCablingTransfer" == "1" ];
    then 
    o2otrans="#o2o_all"
else
    [ "$doPedNoiseTransfer" == "1" ]    && o2otrans="#o2o_pednoi"
    [ "$doFedCablingTransfer" == "1" ] && o2otrans="#o2o_cabl"
fi


VerifyIfO2ONeeded

cfg_file=${test_area}/SiStripO2O_Run_${Run}.cfg


echo DBfile $DBfile
echo DBcatalog $DBcatalog

boolDebug=false
[ "$Debug" == "1" ] && boolDebug=true

[ "$Run" -gt "1" ] && let prevRun=${Run}-1

templatefile=${CMSSW_BASE}/src/OnlineDB/SiStrip020/scripts/template_SiStripO2O.cfg 
[ ! -e $templatefile ] && templatefile=${CMSSW_BASE}/src/CondTools/SiStrip/scripts/template_SiStripO2O.cfg 
[ ! -e $templatefile ] && templatefile=${CMSSW_RELEASE_BASE}/src/OnlineDB/SiStrip020/scripts/template_SiStripO2O.cfg 
[ ! -e $templatefile ] && echo "ERROR: expected template file doesn't exist both in your working area and in release area. Please fix it." && exit

cat $templatefile | sed -e "s@#${geometry}@@g" -e "s#insert_DBfile#$DBfile#g" -e "s#insert_DBcatalog#$DBcatalog#g"  -e "s#insert_IOV#${prevRun}#" -e "s#insert_appendflag#${append}#g" -e "s@#appendMode_${append}@@g" \
-e "s#insert_ConfigDbFull#${ConfigDbUser}/${ConfigDbPasswd}@${ConfigDbPath}#" -e "s#insert_ConfigDbUser#${ConfigDbUser}#g" -e "s#insert_ConfigDbPasswd#${ConfigDbPasswd}#g" -e "s#insert_ConfigDbPath#${ConfigDbPath}#g"	-e "s#insert_ConfigDbPartition#${ConfigDbPartition}#g" -e "s#insert_ConfigDbMajorVersion#${ConfigDbMajorVersion}#g" -e "s#insert_ConfigDbMinorVersion#${ConfigDbMinorVersion}#g" \
-e "s#insert_tagPN#${tagPN}#g"  -e "s#insert_tagCab#${tagCab}#g" \
-e "s#insert_doPedNoiseTransfer#$doPedNoiseTransfer#" -e "s#insert_doFedCablingTransfer#$doFedCablingTransfer#" -e "s#insert_Debug#$Debug#" -e "s@${o2otrans}@@"> ${cfg_file}



echo -e "\ncmsRun ${cfg_file}"
cmsRun ${cfg_file} > ${test_area}/SiStripO2O_${Run}.out

if [ "$?" == 0 ]; then
    echo -e "$0 $@ \n" >> .sistripO2O.log
    echo -e "$ConfigDb \t $ConfigDbPartition \t ${ConfigDbMajorVersion}.${ConfigDbMinorVersion} \t $CondDb \t $tag \t $Run \t $ConfigDbRunMode" >> .SiStripO2OTable.log
fi