#!/bin/sh

echo " Folders available in /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OnlineData/original/:"
ls /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OnlineData/original/

if (( "$#" != "3" ))
    then
    echo "Input parameters needed: <runs folder (among given above)> <runMin> <runMax>";
    exit;
fi

BASEDIR=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OnlineData/original/$1
#00017xxxx

RUNMIN=$2
RUNMAX=$3

ls $BASEDIR/*/DQM_V00*_SiStrip_* > lsafs.dat
awk 'BEGIN { FS = "/" } ; { print $13 }' lsafs.dat > lsfiles.dat
rm lsafs.dat
sed 's/DQM_V0001_SiStrip_R000//' lsfiles.dat > lsruns
rm lsfiles.dat
echo "./extractErrorsvsTime $BASEDIR/ $RUNMIN $RUNMAX " `sed 's/\.root//' lsruns | wc -l` `sed 's/\.root//' lsruns | awk '{ORS=" "}{print $0}'`
# > lsruns.dat
rm lsruns

