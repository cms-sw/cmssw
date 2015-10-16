#!/bin/bash

#export PATH=/afs/cern.ch/cms/common:${PATH}
if [[ "$#" == "0" ]]; then
    echo "usage: 'moduleOccupancyPlots.sh RootFileDirectory Dataset RunNumber ModuleListFile UserCertFile UserKeyFile'";
    exit 1;
fi

nnn=`echo $3 | awk '{print substr($0,0,4)}'` 
curl -k --cert $5 --key $6 -X GET 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/'$1'/'$2'/000'${nnn}'xx/' > index.html
    dqmFileNames=`cat index.html | grep $3 | egrep "_DQM.root|_DQMIO.root" | egrep "Prompt|Express" | sed 's/.*>\(.*\)<\/a.*/\1/' `
    dqmFileName=`expr "$dqmFileNames" : '\(DQM[A-Za-z0-9_/.\-]*root\)'`
    echo ' dqmFileNames = '$dqmFileNames
    echo ' dqmFileName = ['$dqmFileName']'
    curl -k --cert $5 --key $6 -X GET https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/$1/$2/000${nnn}xx/${dqmFileName} > /tmp/${dqmFileName}

moduleOccupancyPlots /tmp/${dqmFileName} $4 4 "$2_" "_run_$3"