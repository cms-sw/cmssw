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
export TNS_ADMIN=/afs/cern.ch/project/oracle/admin

echo cmscond_bootstrap_detector.pl --offline_connect oracle://$CondDb/CMS_COND_STRIP --auth $auth STRIP

cmscond_bootstrap_detector.pl --offline_connect oracle://$CondDb/CMS_COND_STRIP --auth $auth STRIP

