#!/bin/sh

echo " Folders available in /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OnlineData/original/:"
ls /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OnlineData/original/

if (( "$#" != "5" ))
    then
    echo "Input parameters needed: <full path to runs folder (for afs among given above)> <runMin> <runMax> <file field number for awk (13 for afs)> <execute=0 or 1>";
    exit;
fi

#BASEDIR=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OnlineData/original/$1
BASEDIR=$1
#00017xxxx

RUNMIN=$2
RUNMAX=$3
FILEFIELD=$4
EXECSCRIPT=$5

ls $BASEDIR/DQM_V0001_SiStrip_* > lsafs.dat
if (( "$?" != "0" ))
    then
    ls $BASEDIR/*/DQM_V0001_SiStrip_* > lsafs.dat
fi

awk -v var=$FILEFIELD 'BEGIN { FS = "/" } ; { print $var }' lsafs.dat > lsfiles.dat
rm lsafs.dat
sed 's/DQM_V0001_SiStrip_R000//' lsfiles.dat > lsruns
rm lsfiles.dat

if (( "$EXECSCRIPT" == "1" )); then
    echo "Executing command:"
    ./extractErrorsvsTime $BASEDIR/ $RUNMIN $RUNMAX `sed 's/\.root//' lsruns | wc -l` `sed 's/\.root//' lsruns | awk '{ORS=" "}{print $0}'`
else
    echo "./extractErrorsvsTime $BASEDIR/ $RUNMIN $RUNMAX " `sed 's/\.root//' lsruns | wc -l` `sed 's/\.root//' lsruns | awk '{ORS=" "}{print $0}'`
# > lsruns.dat
fi

rm lsruns
