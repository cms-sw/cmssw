#!/bin/sh

#//Get lastIOV from OfflineDB//
function getLastIOV(){
    tag=$1
    if [ "$2" == "devdb10" ]; then
	lastIOV=`cmscond_list_iov -c oracle://devdb10/CMS_COND_STRIP -u CMS_COND_STRIP -p w3807dev -f "relationalcatalog_oracle://devdb10/CMS_COND_GENERAL" -t $1 | grep DB | awk '{print $1}' | tail -1`
    else
	lastIOV=`cmscond_list_iov -c oracle://orcon/CMS_COND_STRIP -u CMS_COND_STRIP_R -p R2106xon -f "relationalcatalog_oracle://orcon/CMS_COND_GENERAL" -t $1 | grep DB | awk '{print $1}' | tail -1`
    fi
}

function getFedVersionFromRunSummaryTIF(){
#    echo  "wget -q -r http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$1&RUN_END=$1&TEXT=1&DB=omds -O lastiov.txt"
    wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$1&RUN_END=$1&TEXT=1&DB=omds" -O lastiov.txt

    #Verify query integrity
    if [ `grep -c "STOPTIME" lastiov.txt` != 1 ]; then
	echo -e "ERROR: RunSummaryTIF provided a strange output"
	cat lastiov.txt	
	rm -f lockFile
	exit
    fi

    FEDVersion_LastIOV=`cat lastiov.txt | awk -F"\t" '{if (NR>1) print $11}'` 
}

function CheckIOV(){

    Run=$1
    
    #//Verify that FEDVersion is not NULL
    FEDVersion_Run=`grep $Run AddedRuns | awk '{print $3}'`
    #echo ${FEDVersion_Run}
    [ "${FEDVersion_Run}" == "" ] && return 11  #//FedVersion NULL: perhaps you are asking for a not existing runNumber 

    if [ "$lastIOV" == "" ]; then
    #//tag $tagPN not found in orcon, check failed//
	return 1
    fi

    if [ "$lastIOV" -ge "$Run" ]; then
    #//tag $tagPN found in orcon, Run inside a closed IOV, check successful//
	return 0
    fi

    [ "${FEDVersion_LastIOV}" == "" ] && return 12  #//FedVersion NULL: perhaps you are asking for a not existing runNumber 
    
    #//Check FEDVersion of $Run and $lastIOV//
    #echo ${FEDVersion_lastIOV}
    [ "${FEDVersion_LastIOV}" == "${FEDVersion_Run}" ] && return 0
    return 3
}

function GetPhysicsRuns(){
    
    #[ ! -e lastRun ] && echo 0 >> lastRun
    #lastRunNb=`tail -1 lastRun`
    #let nextRun=$lastRunNb+1
    #nextRun=6000
    #echo nextRun $nextRun

    [ -e AddedRuns ] && rm -f AddedRuns

    #// Get List of PHYSIC RUNS
    wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$nextRun&RUN_END=1000000000&RUNMODE=PHYSIC&TEXT=1&DB=omds" -O physicsRuns.txt  
    #Verify query integrity
    if [ `grep -c "STOPTIME" physicsRuns.txt` != 1 ]; then
	echo -e "ERROR: RunSummaryTIF provided a strange output for physicsRuns"
	cat physicsRuns.txt
	rm -f lockFile
	exit
    fi

    #// Get List of LATENCY RUNS
    wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$nextRun&RUN_END=1000000000&RUNMODE=LATENCY&TEXT=1&DB=omds" -O latencyRuns.txt
    if [ `grep -c "STOPTIME" latencyRuns.txt` != 1 ]; then
	echo -e "ERROR: RunSummaryTIF provided a strange output for latencyRuns"
	cat latencyRuns.txt
	rm -f lockFile
	exit
    fi

    cat physicsRuns.txt latencyRuns.txt | sort -r | awk -F"\t" '{if (NR>2) print $1" "$2" "$11" "$13}' | sort >> AddedRuns
}


############
##  MAIN  ##
############

scriptDir=`dirname $0`
cd $scriptDir

[ -e lockFile ] && exit 
touch lockFile

CondDB=""
[ "$1" != "" ] && CondDB=$1

nextRun=5000
[ "$3" != "" ] && nextRun=$3 

export lastIOV
export FEDVersion_LastIOV
export nextRun

eval `scramv1 runtime -sh`

rm -f RunToBeSubmitted_${CondDB}.tmp  RunToDoO2O_${CondDB}.tmp
touch  RunToBeSubmitted_${CondDB}  RunToDoO2O_${CondDB}

GetPhysicsRuns

oldFedVer=0
oldtagPN=""
for Run in `cat AddedRuns | awk '{print $1}'`
  do
  echo "[MonitorO2O.sh] Looking at run $Run"
  
  ConfigDb=`grep $Run AddedRuns | awk '{print $4}'`   

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

  ver="v1"
  [ "$2" != "" ] && ver=$2
  
  vTag=${tag}_${ver}
  tagPN=SiStripPedNoise_${vTag}_p

  FedVer=`grep $Run AddedRuns | awk '{print $3}'`

  #echo CheckIOV $Run $tagPN
  if [ "${oldtagPN}" != "${tagPN}" ]; then
      oldtagPN=${tagPN}
      getLastIOV $tagPN $CondDB
      FEDVersion_LastIOV=""
      [ "$lastIOV" != "" ] && getFedVersionFromRunSummaryTIF $lastIOV
      echo -e "[MonitorO2O.sh] DB=$CondDB \ttag=$vTag \tlastIOV=$lastIOV \tFEDVersion_LastIOV=${FEDVersion_LastIOV}"
  fi
  
  CheckIOV $Run 
  status=$?

  #echo status $status
  if [ "$status" == "0" ];
      then
      echo -e "$Run \t ${FedVer} \t $CondDB \t $vTag">> RunToBeSubmitted_${CondDB}.tmp
  elif [ $status -lt 10 ]; then
      if [ "${FedVer}" != "${oldFedVer}" ] ; then
	  oldFedVer=${FedVer}
	  echo -e "$Run \t $status \t $CondDB \t $vTag \t ${FedVer} \t $ConfigDb" >> RunToDoO2O_${CondDB}.tmp
      fi
  else
      # case of error 11 and 12 in CheckIOV
      echo "[MonitorO2O.sh] ERROR: CheckIOV answer $status"
      rm -f lockFile
      exit
  fi
done

#if [ `cat AddedRuns | wc -l ` != "0" ]; then
if [ -e RunToBeSubmitted_${CondDB}.tmp ]; then
    if [ "`diff -q RunToBeSubmitted_${CondDB} RunToBeSubmitted_${CondDB}.tmp`" != "" ]; then
	mv -f RunToBeSubmitted_${CondDB}.tmp RunToBeSubmitted_${CondDB}
	cat RunToBeSubmitted_${CondDB} | mail -s "MonitorO2O cron: Submit Runs" "domenico.giordano@cern.ch noeding@fnal.gov Nicola.Defilippis@ba.infn.it sdutta@mail.cern.ch"
	#cat RunToBeSubmitted_${CondDB} | mail -s "MonitorO2O cron: Submit Runs" "domenico.giordano@cern.ch sdutta@mail.cern.ch"
	#echo "There are jobs to be Submitted" | mail -s " " 00393402949274@sms.switch.ch
    fi
else 
    rm -f RunToBeSubmitted_${CondDB}
    touch RunToBeSubmitted_${CondDB}
fi
#    if [ `cat RunToDoO2O.tmp | wc -l` != "0" ] && [ "`diff -q RunToDoO2O RunToDoO2O.tmp`" != "" ]; then
if [ -e RunToDoO2O_${CondDB}.tmp ]; then
    if [ "`diff -q RunToDoO2O_${CondDB} RunToDoO2O_${CondDB}.tmp`" != "" ]; then
	mv -f RunToDoO2O_${CondDB}.tmp RunToDoO2O_${CondDB}
    fi
else
    rm -f  RunToDoO2O_${CondDB}
    touch  RunToDoO2O_${CondDB}
fi
if [ `cat RunToDoO2O_${CondDB} | wc -l` != "0" ]; then
    cat RunToDoO2O_${CondDB} | mail -s "MonitorO2O cron: Do o2o" domenico.giordano@cern.ch
#    echo "There is o2o to be done" | mail -s " " 00393402949274@sms.switch.ch
fi
#fi
#fi

rm -f lockFile