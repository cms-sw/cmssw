#!/bin/sh

cd `dirname $0`
basePath=`pwd`
source /nfshome0/cmssw2/scripts/setup.sh
cd /raid/cmssw/Development/O2O/CMSSW_2_0_8/src/
eval `scramv1 runtime -sh`
cd $basePath
export TNS_ADMIN=/nfshome0/xiezhen/conddb 

onlineDB="cms_trk_tkcc/tkcc2008@cms_omds_lb"

sqlplus -S -M "HTML ON " $onlineDB < discoverPedestals.sql | awk 'BEGIN{newline=0; stringa=""} $0~/<*th.*>/{newline=0} $0~/<tr>/{newline=1;if(stringa!="") print stringa; stringa=""} $0~/<\/tr>/{newline=0;} $0!~/<?td.*>/{if(newline && $0!="<tr>"){stringa=sprintf("%s %s",stringa,$0)}} END{print "99999999999999 LATENCY"}' | sort -n| while read line;
do
vec=($line)

echo -e "----------------\n${vec[@]}\n----------------"
done
