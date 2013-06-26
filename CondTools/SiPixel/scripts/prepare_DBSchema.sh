#!/bin/sh

function usage () {
    
    echo -e "\n[usage] `basename $0` [options]"
    echo -e " -CondDb=<devdb10>, <orcon> (no default )"
    echo -e " -auth=<coral_auth_path> (default is ${CORAL_AUTH_PATH}/authentication.xml )"

    
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


#################
## MAIN
#################

export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb

[ `echo $@ | grep -c "\-help"` = 1 ] && usage;

getParameter CondDb    $@ ""
getParameter auth      $@ ${CORAL_AUTH_PATH}/authentication.xml

if [ "$CondDb" != "devdb10" ] && [ "$CondDb" != "orcon" ];
    then
    echo -e "\nERROR: wrong options"
    usage
fi
eval `scramv1 runtime -sh`


#echo $CMSSW_BASE/src/CondTools/OracleDBA/scripts/cmscond_bootstrap_detector.pl --offline_connect oracle://$CondDb/CMS_COND_PIXEL --auth $auth PIXEL
#
# Questo e' per devdb10/orcon
#
#$CMSSW_BASE/src/CondTools/OracleDBA/scripts/cmscond_bootstrap_detector.pl --offline_connect=oracle://cms_orcoff_int2r/CMS_COND_PIXEL --auth /afs/cern.ch/cms/DB/conddb/devowner/authentication.xml --catalog=relationalcatalog_oracle://cms_orcoff_int2r/CMS_COND_GENERAL PIXEL --debug
#
# Qusto e' per SQLite
#
#echo $CMSSW_BASE/src/CondTools/OracleDBA/scripts/cmscond_bootstrap_detector.pl --offline_connect sqlite_file:prova.db --auth $auth PIXEL
$CMSSW_BASE/src/CondTools/OracleDBA/scripts/cmscond_bootstrap_detector.pl --offline_connect sqlite_file:prova.db --auth $auth PIXEL --debug
