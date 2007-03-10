#!/bin/sh

#//Get lastIOV from OfflineDB//
function getLastIOV(){
    tag=$1
#    lastIOV=`cmscond_list_iov -c oracle://devdb10/CMS_COND_STRIP -u CMS_COND_STRIP -p w3807dev -f "relationalcatalog_oracle://devdb10/CMS_COND_GENERAL" -t $1 | grep DB | awk '{print $1}' | tail -1`
    lastIOV=`cmscond_list_iov -c oracle://orcon/CMS_COND_STRIP -u CMS_COND_STRIP_R -p R2106xon -f "relationalcatalog_oracle://orcon/CMS_COND_GENERAL" -t $1 | grep DB | awk '{print $1}' | tail -1`
}

function getFedVersionFromRunSummaryTIF(){

    wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$1&RUN_END=$1&TEXT=1&DB=omds" -O out.txt

    FEDVersion_lastIOV=`cat out.txt | awk -F"\t" '{if (NR>1) print $11}'` 
}

function CheckIOV(){

    Run=$1
    
    #//Verify that FEDVersion is not NULL
    FEDVersion_Run=`grep $Run AddedRuns | awk '{print $3}'`
    #echo ${FEDVersion_Run}
    [ "${FEDVersion_Run}" == "" ] && return 2  #//FedVersion NULL: perhaps you are asking for a not existing runNumber 

    if [ "$lastIOV" == "" ]; then
    #//tag $tagPN not found in orcon, check failed//
	return 1
    fi
    if [ "$lastIOV" -ge "$Run" ]; then
    #//tag $tagPN found in orcon, Run inside a closed IOV, check successful//
	return 0
    fi
    
    #//Check FEDVersion of $Run and $lastIOV//
    #echo ${FEDVersion_lastIOV}
    [ "${FEDVersion_lastIOV}" == "${FEDVersion_Run}" ] && return 0
    return 3
}

function GetPhysicsRuns(){
    
    #[ ! -e lastRun ] && echo 0 >> lastRun
    #lastRunNb=`tail -1 lastRun`
    #let nextRun=$lastRunNb+1
    nextRun=0
    echo nextRun $nextRun

    [ -e AddedRuns ] && rm -f AddedRuns

    #// Get List of PHYSIC RUNS
    wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$nextRun&RUN_END=1000000000&RUNMODE=PHYSIC&TEXT=1&DB=omds" -O physicsRuns.txt  
    #// Get List of LATENCY RUNS
    wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$nextRun&RUN_END=1000000000&RUNMODE=LATENCY&TEXT=1&DB=omds" -O latencyRuns.txt

    cat physicsRuns.txt latencyRuns.txt | sort -r | awk -F"\t" '{if (NR>2) print $1" "$2" "$11" "$13}' | sort >> AddedRuns
}


############
##  MAIN  ##
############

scriptDir=`dirname $0`
cd $scriptDir

[ -e lockFile ] && exit 
touch lockFile

eval `scramv1 runtime -sh`


export lastIOV
export Fedversion_Lastiov

touch  RunToBeSubmitted.bkp  RunToDoO2O.bkp
[ -e RunToBeSubmitted ] && mv -f RunToBeSubmitted RunToBeSubmitted.bkp
[ -e RunToDoO2O ] && mv -f RunToDoO2O RunToDoO2O.bkp
touch  RunToBeSubmitted  RunToDoO2O

GetPhysicsRuns

oldFedVer=0
oldtagPN=""
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

  FedVer=`grep $Run AddedRuns | awk '{print $3}'`

  #echo CheckIOV $Run $tagPN
  if [ "${oldtagPN}" != "${tagPN}" ]; then
      oldtagPN=${tagPN}
      getLastIOV $tagPN
      getFedVersionFromRunSummaryTIF $lastIOV
  fi
  
  CheckIOV $Run 
  status=$?

  #echo status $status
  if [ "$status" == "0" ];
      then
      echo $Run $vTag >> RunToBeSubmitted
  else
      if [ "${FedVer}" != "${oldFedVer}" ] ; then
	  oldFedVer=${FedVer}
	  echo $Run $vTag $status >> RunToDoO2O
      fi
  fi
done

if [ `cat AddedRuns | wc -l ` != "0" ]; then
    if [ "`diff -q RunToBeSubmitted RunToBeSubmitted.bkp`" != "" ]; then
	cat RunToBeSubmitted | mail -s "MonitorO2O cron: Submit Runs " domenico.giordano@cern.ch
#	echo "" | mail -s "There are jobs to be Submitted" 00393402949274@sms.switch.ch
    fi
    
    if [ "`diff -q RunToDoO2O RunToDoO2O.bkp`" != "" ]; then
	cat RunToDoO2O | mail -s "MonitorO2O cron: Do o2o" domenico.giordano@cern.ch
#	echo "" | mail -s "There is o2o to be done" 00393402949274@sms.switch.ch
    fi
fi

rm -f lockFile