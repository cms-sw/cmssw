#!/bin/bash

cd `dirname $0`
basePath=`pwd`
echo basePath $basePath

source /nfshome0/cmssw2/scripts/setup.sh
cd /raid/cmssw/Development/O2O/CMSSW_2_1_0_pre3/src/
eval `scramv1 runtime -sh`
cd $basePath

preTag=TKCC_21X
postTag=test8
Tag=${preTag}_${postTag}
authPath=/nfshome0/xiezhen/conddb/
#authPath=/afs/cern.ch/cms/DB/conddb
onlineDB="cms_trk_tkcc/tkcc2008@cms_omds_lb"
connectString="oracle://cms_orcon_prod/CMS_COND_21X_STRIP"
#connectString="oracle://cms_orcoff_int2r/CMS_COND_STRIP"
#connectString="sqlite_file:dbfile.db"
#logdb="sqlite_file:log.db"
logdb="oracle://omds/CMS_POP_CON"
storeFile="/raid/cmssw/Development/O2O/store/doneRuns_${preTag}_${postTag}"
doExit="false"

touch $storeFile

log=log_at_P5_$Tag
#rm -rf $log
mkdir -p $log

export TNS_ADMIN=/nfshome0/xiezhen/conddb/

#sqlplus -S -M "HTML ON " $onlineDB < OMDSQuery.sql

sqlplus -S -M "HTML ON " $onlineDB < OMDSQuery.sql | awk 'BEGIN{newline=0; stringa=""} $0~/<*th.*>/{newline=0} $0~/<tr>/{newline=1;if(stringa!="") print stringa; stringa=""} $0~/<\/tr>/{newline=0;} $0!~/<?td.*>/{if(newline && $0!="<tr>"){stringa=sprintf("%s %s",stringa,$0)}}' | sort -n| while read line;
do
vec=($line)

echo ${vec[@]}

runNb=${vec[0]}
mode=${vec[1]}
partition=${vec[2]}
fecV=${vec[3]}
fedV=${vec[4]}
cabV=${vec[5]}
dcuV=${vec[6]}

#echo $runNb $partition $fecV $fedV $cabV

fedVMaj=`echo $fedV | cut -d "." -f1`
fedVmin=`echo $fedV | cut -d "." -f2`

fecVMaj=`echo $fecV | cut -d "." -f1`
fecVmin=`echo $fecV | cut -d "." -f2`

cabVMaj=`echo $cabV | cut -d "." -f1`
cabVmin=`echo $cabV | cut -d "." -f2`

dcuVMaj=`echo $dcuV | cut -d "." -f1`
dcuVmin=`echo $dcuV | cut -d "." -f2`

if [ "$mode" == "PEDESTAL" ]; then
    runVer="$partition $cabV $fedV $dcuV"
else
    runVer="$partition $cabV XXX $dcuV"
fi
runString="$runNb $runVer"
echo -n "$runString  --  "
if grep -q "$runString" $storeFile ; then
    echo O2O transfer already performed or not necessary
else
      # O2O not done, let's check if it is necessary
    if tail -n1 $storeFile | grep -qe "$runVer" ; then
	echo Fed and partition not changed since last IOV was opened. O2O Not necessary
    elif [ 0`tail -n1 $storeFile | awk '{print $1}'` -gt $runNb ]; then 
	echo RunNumber $runNb violates append mode. last uploaded run is `tail -n1 $storeFile | awk '{print $1}'` 
	echo -e "\n Please start the o2o with a new offline tag name in file $basePath/`basename $0`\nCurrent tag is $Tag" 
	exit
    else
	echo Performing O2O...
    fi
fi

[[ "$doExit" != "false" ]] && break

done

echo ""
