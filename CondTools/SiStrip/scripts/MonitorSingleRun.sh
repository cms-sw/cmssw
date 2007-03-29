#!/bin/sh

#//Get lastIOV from OfflineDB//
function getLastIOV(){
    tag=$1
    echo "cmscond_list_iov -c frontier://cmsfrontier.cern.ch:8000/FrontierInt/cms_cond_strip  -f file:stripcatalog.xml -t $1 "
    lastIOV=`cmscond_list_iov -c frontier://cmsfrontier.cern.ch:8000/FrontierInt/cms_cond_strip  -f file:stripcatalog.xml -t $1 | grep DB | awk '{print $1}' | tail -1`
}

function getFedVersionFromRunSummaryTIF(){
#    echo  "wget -q -r http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$1&RUN_END=$1&TEXT=1&DB=omds -O lastiov.txt"
    wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$1&RUN_END=$1&TEXT=1&DB=omds" -O lastiov.txt

    #Verify query integrity
    if [ `grep -c "STOPTIME" lastiov.txt` != 1 ]; then
	echo -e "ERROR: RunSummaryTIF provided a strange output"
	cat lastiov.txt	
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



############
##  MAIN  ##
############
Run=0
[ "$1" != "" ] && Run=$1
ver="v1"
[ "$2" != "" ] && ver=$2

scriptDir=`dirname $0`
cd $scriptDir

CondDB=frontier

export lastIOV
export FEDVersion_LastIOV

eval `scramv1 runtime -sh`

#// Get List of PHYSIC RUNS
echo wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$1&RUN_END=$1&TEXT=1&DB=omds" -O physicsRuns.txt  
wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$1&RUN_END=$1&TEXT=1&DB=omds" -O physicsRuns.txt  

 #Verify query integrity
if [ `grep -c "STOPTIME" physicsRuns.txt` != 1 ]; then
    echo -e "ERROR: RunSummaryTIF provided a strange output for physicsRuns"
    cat physicsRuns.txt
    exit
fi
cat physicsRuns.txt | awk -F"\t" '{if (NR>2) print $1" "$2" "$11" "$13}' | sort >> AddedRuns

for Run in `cat physicsRuns.txt | awk '{print $1}'`
  do
  echo "[MonitorSingleRun.sh] Looking at run $Run"
  
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
  
  vTag=${tag}_${ver}
  tagPN=SiStripPedNoise_${vTag}_p

  FedVer=`grep $Run AddedRuns | awk '{print $3}'`

  echo CheckIOV $Run $tagPN
  getLastIOV $tagPN $CondDB
  FEDVersion_LastIOV=""
  [ "$lastIOV" != "" ] && getFedVersionFromRunSummaryTIF $lastIOV
  echo -e "[MonitorSingleRun.sh] DB=$CondDB \ttag=$vTag \tlastIOV=$lastIOV \tFEDVersion_LastIOV=${FEDVersion_LastIOV}"
    
  CheckIOV $Run 
  status=$?

  #echo status $status
  if [ "$status" == "0" ];
      then
      echo -e "You can submit the run: $Run \t ${FedVer} \t $CondDB \t $vTag"
  elif [ $status -lt 10 ]; then
      if [ "${FedVer}" != "${oldFedVer}" ] ; then
	  oldFedVer=${FedVer}
	  echo -e "Need O2O for $Run \t $status \t $CondDB \t $vTag \t ${FedVer} \t $ConfigDb" 
      fi
  else
      # case of error 11 and 12 in CheckIOV
      echo "[MonitorSingleRun.sh] ERROR: CheckIOV answer $status"
      exit
  fi
done

