#!/bin/sh

function GetPhysicsRuns(){
    
    [ ! -e lastRun ] && echo 0 >> lastRun
    lastRunNb=`tail -1 lastRun`
    let nextRun=$lastRunNb+1

    echo nextRun $nextRun

    [ -e AddedRuns ] && rm -f AddedRuns

    #// Get List of PHYSIC RUNS
    wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$nextRun&RUN_END=1000000000&RUNMODE=PHYSIC&TEXT=1&DB=omds" -O physicsRuns.txt  
    #// Get List of LATENCY RUNS
    wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$nextRun&RUN_END=1000000000&RUNMODE=LATENCY&TEXT=1&DB=omds" -O latencyRuns.txt

    cat physicsRuns.txt latencyRuns.txt | sort -r | awk -F"\t" '{if (NR>2) print $1" "$2" "$11}' | sort >> AddedRuns

    if [ -e RunList ]; then
	mv -f RunList RunList.tmp
	cat RunList.tmp AddedRuns | sort >> RunList
    else
	sort AddedRuns >> RunList
    fi
    tail -1 AddedRuns | awk '{print $1}' >> lastRun
    
    rm -f *.tmp
}


############
##  MAIN  ##
############

[ -e lockFile ] && exit 
touch lockFile

scriptDir=`dirname $0`
cd $scriptDir


GetPhysicsRuns

. ./CheckIOV.sh 

for Run in `cat AddedRuns | awk '{print $1}'`
  do
  echo Looking at run $Run
  
  tag=""
  ConfigDbPartition=`grep $Run AddedRuns | awk '{print $2}'`   
  if [ `echo $ConfigDbPartition | grep -c -i TIBTOBTEC` == '1' ]; then
      tag=TIBTOBTEC
  elif [ `echo $ConfigDbPartition | grep -c -i TIBTOB` == '1' ]; then
      tag=TIBTOB
  elif  [ `echo $ConfigDbPartition | grep -c -i TIB`    == '1' ]; then
      tag=TIB
  elif  [ `echo $ConfigDbPartition | grep -c -i TOB`    == '1' ]; then
      tag=TOB
  elif  [ `echo $ConfigDbPartition | grep -c -i TEC`    == '1' ]; then
      tag=TEC
  fi
  
  [ "$tag" == "" ] && continue

  vTag=${tag}_v1
  tagPN=SiStripPedNoise_${vTag}_p

  #echo CheckIOV $Run $tagPN
  CheckIOV $Run $tagPN
  status=$?
  #echo status $status
  if [ "$status" == "0" ];
      then
      echo $Run $vTag >> RunToBeSubmitted
  else
      echo $Run $vTag $status >> RunToDoO2O
  fi
done

if [ "`wc -l AddedRuns`" != "0" ]; then
    if [ "`wc -l RunToBeSubmitted`" != "0" ]; then
	cat RunToBeSubmitted | mail -s "MonitorO2O cron: Submit Runs " domenico.giordano@cern.ch
	echo "" | mail -s "There are jobs to be Submitted" 00393402949274@sms.switch.ch
    fi
    
    if [ "`wc -l RunToDoO2O`" != "0" ]; then
	cat RunToDoO2O | mail -s "MonitorO2O cron: Do o2o" domenico.giordano@cern.ch
	echo "" | mail -s "There is o2o to be done" 00393402949274@sms.switch.ch
    fi
fi

rm -f lockFile