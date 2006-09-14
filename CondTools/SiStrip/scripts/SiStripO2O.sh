#!/bin/sh

function usage () {
    echo -e "\n[usage]\n SiStripO2O.sh [options]"
    echo -e " -IOV=<runNb> (default is $default_IOV )"
    echo -e " -ConfigDb=<user/passwd@path> (default is ${default_ConfigDb} )"
    echo -e " -ConfigDbVersion=<Major.Minor> (default is ${default_ConfigDb} )"
    echo -e " -ConfigDbPartition=<partitionName> (default is ${default_ConfigDbPartition} )"
    echo -e " -doPedNoiseTransfer (default is $default_doPedNoiseTransfer )"
    echo -e " -doFedCablingTransfer (default is $default_doFedCablingTransfer )" 
    echo -e " -CondDb=<sqlite>, <devdb10>, <orcon> (default is sqlite)"
    echo -e " -sqliteDb=<path_name> (default is /tmp/$USER/o2o/dummy_<IOV>.db)"
    echo -e " -sqliteCatalog=<path_name> (default is /tmp/$USER/o2o/dummy_<IOV>.xml)"
    echo -e " -firstUpload (otherwise works in append mode) "

    
    exit
}

function getParameter(){
    what=$1
    eval $what=\$$#
    shift
    where=$@
    if [ `echo $where | grep -c "\-$what="` = 1 ]; then
        eval $what=`echo $where | awk -F"${what}=" '{print $2}' | awk '{print $1}'`
    elif [ `echo $where | grep -c "\-$what "` = 1 ]; then
	eval $what=1
    fi
}

function settings (){
    export TNS_ADMIN=/afs/cern.ch/project/oracle/admin
    export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb
    export test_area=/tmp/$USER/o2o

    default_IOV=1
    default_doPedNoiseTransfer=0
    default_doFedCablingTransfer=0
    default_ConfigDb=cms_mtcc_sitracker/cms_mtcc@omds
    default_ConfigDbVersion=8.189
    default_ConfigDbPartition="MTCC_DEMO"
    default_CondDb="sqlite"
    default_firstUpload=0
    
    getParameter help $@ 0
    [ "$help" = 1 ] && usage

    getParameter doPedNoiseTransfer   $@ ${default_doPedNoiseTransfer}
    getParameter doFedCablingTransfer $@ ${default_doFedCablingTransfer}
    getParameter ConfigDb             $@ ${default_ConfigDb}
    getParameter ConfigDbVersion      $@ ${default_ConfigDbVersion}
    getParameter ConfigDbPartition    $@ ${default_ConfigDbPartition}
    getParameter IOV                  $@ ${default_IOV}
    getParameter CondDb               $@ ${default_CondDb}
    getParameter firstUpload          $@ ${default_firstUpload} 

    default_sqliteDb=${test_area}/dummy_${IOV}.db
    default_sqliteCatalog=${test_area}/dummy_${IOV}.xml
    getParameter sqliteDb             $@ ${default_sqliteDb}
    getParameter sqliteCatalog        $@ ${default_sqliteCatalog}


    [ ! -e ${sqliteDb} ] && firstUpload=1

    append=1
    [ "$firstUpload" = 1 ] && append=0
    export append
    
    ConfigDbUser=`echo ${ConfigDb}| awk -F'/' '{print $1}'`
    ConfigDbPasswd=`echo ${ConfigDb}| awk -F'/' '{print $2}' | awk -F'@' '{print $1}'`
    ConfigDbPath=`echo ${ConfigDb}| awk -F'@' '{print $2}'`
    ConfigDbMajorVersion=`echo ${ConfigDbVersion}| awk -F'.' '{print $1}'`
    ConfigDbMinorVersion=`echo ${ConfigDbVersion}| awk -F'.' '{print $2}'`

    echo -e " -IOV=$IOV"
    echo -e " -ConfigDb=${ConfigDb}"
    echo -e " -ConfigDbVersion=${ConfigDbVersion}"
    echo -e " -ConfigDbPartition=${ConfigDbPartition}"
    echo -e " -CondDb=$CondDb"
    echo -e " -firstUpload $firstUpload"
    echo -e " -doPedNoiseTransfer $doPedNoiseTransfer"
    echo -e " -doFedCablingTransfer $doFedCablingTransfer"
    if [ "$CondDb" = "sqlite" ]; then
	echo -e " -sqliteDb=${sqliteDb}"
	echo -e " -sqliteCatalog=${sqliteCatalog}"
    fi
    echo " "
}

#################
## MAIN
#################


settings "$@"


[ ! -e ${test_area} ] && mkdir -p ${test_area}


eval `scramv1 runtime -sh`

  
if [ "$CondDb" == "devdb10" ]; then
    DBfile="oracle://devdb10/CMS_COND_STRIP"
    DBcatalog="relationalcatalog_oracle://devdb10/CMS_COND_GENERAL"
elif [ "$CondDb" == "orcon" ]; then
    DBfile="oracle://orcon/CMS_COND_STRIP"
    DBcatalog="relationalcatalog_oracle://orcon/CMS_COND_GENERAL"
elif [ "$CondDb" == "sqlite" ]; then
    DBfile="sqlite_file:${sqliteDb}"
    DBcatalog="file:${sqliteCatalog}"

    if [ "${append}" == 0 ] ;
	then 
	rm -f ${sqliteDb}
	rm -f ${sqliteCatalog}
        
	echo "CondTools/OracleDBA/scripts/cmscond_bootstrap_detector.pl --offline_connect $DBfile --auth ${CORAL_AUTH_PATH}/authentication.xml STRIP "
	$CMSSW_BASE/src/CondTools/OracleDBA/scripts/cmscond_bootstrap_detector.pl --offline_connect $DBfile --auth ${CORAL_AUTH_PATH}/authentication.xml STRIP 
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


cfg_file=${test_area}/SiStripO2O_IOV_${IOV}.cfg


echo DBfile $DBfile
echo DBcatalog $DBcatalog

cat template_SiStripO2O.cfg | sed -e "s#insert_DBfile#$DBfile#g" -e "s#insert_DBcatalog#$DBcatalog#g"  -e "s#insert_IOV#${IOV}#" -e "s#insert_appendflag#${append}#g" -e "s@#appendMode_${append}@@g" \
-e "s#insert_ConfigDbUser#${ConfigDbUser}#" -e "s#insert_ConfigDbPasswd#${ConfigDbPasswd}#" -e "s#insert_ConfigDbPath#${ConfigDbPath}#"	-e "s#insert_ConfigDbPartition#${ConfigDbPartition}#" -e "s#insert_ConfigDbMajorVersion#${ConfigDbMajorVersion}#" -e "s#insert_ConfigDbMinorVersion#${ConfigDbMinorVersion}#" \
-e "s#insert_doPedNoiseTransfer#$doPedNoiseTransfer#" -e "s#insert_doFedCablingTransfer#$doFedCablingTransfer#" -e "s@${o2otrans}@@"> ${cfg_file}



echo -e "\ncmsRun ${cfg_file}"
cmsRun ${cfg_file} > ${test_area}/out_o2o_${IOV}


