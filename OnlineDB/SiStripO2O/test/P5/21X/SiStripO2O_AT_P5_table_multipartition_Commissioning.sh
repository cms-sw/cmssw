#!/bin/bash

function DoO2O(){
    echo -n "$runString  --  "
    if grep -q "$runString" $storeFile ; then
	echo O2O transfer already performed or not necessary
    else
      # O2O not done, let's check if it is necessary
	if tail -n1 $storeFile | grep -qe "`echo $runString | cut -d":" -f 2-`" ; then
	    echo Fed and partition not changed since last IOV was opened. O2O Not necessary
	    echo "$runString -- Not Necessary" >> $storeFile
	elif [ 0`tail -n1 $storeFile | awk '{print $1}'` -gt $oldRunNb ]; then
	    echo RunNumber $oldRunNb violates append mode. last uploaded run is `tail -n1 $storeFile | awk '{print $1}'`
	    echo -e "\n Please, to include this run, start the o2o with a new offline tag name in file $basePath/`basename $0`\nCurrent tag is $Tag"
	    echo -e "\n--- Run skipped ----\n"
	    return
	else
	    echo Performing O2O...
	    #echo pedmode $pedmode oldrunNb $oldRunNb oldpedmode $oldPedmode
	    cp template_sistripo2o_multipartition.cfg $log/tmp.cfg
	    counter=0
	    echo $runString | sed -e "s@:Part:@\n@g" | awk '{ if(NR>1) print $0 }'| while read line;
	    do
	      vec=($line)
	      partition=${vec[0]}	      
	      #echo "----------------"
	      #echo ${vec[@]}
	      #echo partition $partition
	      #echo "----------------"

	      let counter++
	      
	      cat  $log/tmp.cfg | sed -e "s#insert_onlineDB#$onlineDB#g" -e "s@insert_Run@$oldRunNb@g" -e "s@insert_ConfigDbPartition_$counter@$partition@g"  -e "s@insert_authPath@$authPath@" -e "s@insert_tag@$Tag@" -e "s@insert_connect_string@$connectString@" -e "s@#DO${oldPedmode}@@" -e "s@#thePart${counter}@@" -e "s@insert_logdb@$logdb@"> $log/tmp1.cfg
	      mv $log/tmp1.cfg $log/tmp.cfg
	    done
	    cp $log/tmp.cfg $log/siStripO2O_$oldRunNb.cfg
	    rm $log/tmp.cfg 

	    if cmsRun $log/siStripO2O_$oldRunNb.cfg > $log/siStripO2O_$oldRunNb.log ; then
		if [ "$oldPedmode" == "PEDESTAL" ]; then 
		    echo "$runString -- CabPed">> $storeFile
		else
		    echo "$runString -- CabOnly">> $storeFile
		fi
	    else
		echo ERROR in O2O performing. EXIT!
		echo check log file `pwd`/$log/siStripO2O_$oldRunNb.log
		doExit="true"
	    fi
	fi
    fi
}

##################################################
######## MAIN
#####################################


cd `dirname $0`
basePath=`pwd`

[ -e "LockFileComm" ] && echo -e "\n Lock File already exists. \nProbably a process is already running. \nOtherwise, please remove the lockFile from $basePath/LockFileComm" && exit
LOCKFILE=`pwd`/LockFileComm
ERRORLOCKFILE="/tmp/o2o-error-`date +%Y%m%d%H%M`"
touch $LOCKFILE
touch $ERRORLOCKFILE 
trap "rm -f $LOCKFILE" EXIT

source /nfshome0/cmssw2/scripts/setup.sh
cd /raid/cmssw/Development/O2O/CMSSW_2_1_0_pre3/src/
eval `scramv1 runtime -sh`
cd $basePath

preTag=TKCC_21X
postTag=Comm_v1
Tag=${preTag}_${postTag}
authPath=/nfshome0/xiezhen/conddb/
#authPath=/afs/cern.ch/cms/DB/conddb
onlineDB="cms_trk_tkcc/tkcc2008@cms_omds_lb"
connectString="oracle://cms_orcon_prod/CMS_COND_21X_STRIP"
#connectString="oracle://cms_orcoff_int2r/CMS_COND_STRIP"
#connectString="sqlite_file:dbfile.db"
#logdb="sqlite_file:log.db"
#logdb="oracle://omds/CMS_POP_CON"
logdb="oracle://cms_orcon_prod/CMS_COND_21X_POPCONLOG"
storeFile="/raid/cmssw/Development/O2O/store/doneRuns_${preTag}_${postTag}"
doExit="false"

touch $storeFile

log=log_at_P5_$Tag
#rm -rf $log
mkdir $log

export TNS_ADMIN=/nfshome0/xiezhen/conddb 

#sqlplus -S -M "HTML ON " $onlineDB < OMDSQuery.sql

oldRunNb=-1
oldPedmode=blablabla
runString=""

sqlplus -S -M "HTML ON " $onlineDB < OMDSQuery_multipartition.sql | awk 'BEGIN{newline=0; stringa=""} $0~/<*th.*>/{newline=0} $0~/<tr>/{newline=1;if(stringa!="") print stringa; stringa=""} $0~/<\/tr>/{newline=0;} $0!~/<?td.*>/{if(newline && $0!="<tr>"){stringa=sprintf("%s %s",stringa,$0)}} END{print "99999999999999 LATENCY"}' | sort -n| while read line;
do
vec=($line)

#echo -e "----------------\n${vec[@]}\n----------------"
#echo $runNb $partition $fecV $fedV $cabV

runNb=${vec[0]}
mode=${vec[1]}
partition=${vec[2]}
fecV=${vec[3]}
fedV=${vec[4]}
cabV=${vec[5]}
dcuV=${vec[6]}

if [ "$mode" == "LATENCY" ] || [ "$mode" == "PHYSIC" ]; then
    runVer=":Part: $partition $cabV $fedV $dcuV"
    pedmode="PEDESTAL"
    continue
elif [ "$mode" == "PEDESTAL" ]; then
    runVer=":Part: $partition $cabV XXX $dcuV"
    pedmode="blablabla"
else
    continue
fi

if [ "$oldRunNb" != "$runNb" ]; then
    if [ "$oldRunNb" != "-1" ]; then
     #No other equal partitions
	#echo "------------------------>" doO2O for $runString 
	DoO2O
    fi
    runString="$runNb $mode"
    oldRunNb=$runNb
    oldPedmode=$pedmode
fi
runString="$runString $runVer"


[[ "$doExit" != "false" ]] && break


echo ""

done

echo ""

rm -f $LOCKFILE
rm -f $ERRORLOCKFILE
