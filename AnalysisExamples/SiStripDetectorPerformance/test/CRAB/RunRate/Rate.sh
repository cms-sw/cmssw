#!/bin/sh

    
function GetPhysicsRuns(){
    
    [ -e AddedRuns ] && rm -f AddedRuns
    
    #// Get List of PHYSIC RUNS
    wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$1&RUN_END=1000000000&RUNMODE=PHYSIC&TEXT=1&DB=omds" -O physicsRuns.tmp
    #Verify query integrity
    if [ `grep -c "STOPTIME" physicsRuns.tmp` != 1 ]; then
        echo -e "ERROR: RunSummaryTIF provided a strange output for physicsRuns"
        cat physicsRuns.tmp
	return
    fi

    wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$1&RUN_END=1000000000&RUNMODE=PHYSIC_ZERO_SUPPRESSION&TEXT=1&DB=omds" -O physicsZSRuns.tmp
    #Verify query integrity
    if [ `grep -c "STOPTIME" physicsZSRuns.tmp` != 1 ]; then
        echo -e "ERROR: RunSummaryTIF provided a strange output for physicsRuns"
        cat physicsZSRuns.tmp
	return
    fi

    rm physicsRuns.txt.tmp
    rm physicsZSRuns.txt.tmp
    [ -e physicsRuns.txt ] && mv physicsRuns.txt physicsRuns.txt.tmp
    cat physicsRuns.tmp physicsZSRuns.tmp | grep -v "STOPTIME" >> physicsRuns.txt.tmp
    sort -n physicsRuns.txt.tmp > physicsRuns.txt
}

function GetListOfEvents(){
physicsRunsFile=$1
rawdatasetStatFile=$2
recodatasetStatFile=$3
recodatasetStatFileFNAL=$4

rm RateFull.html
rm RateShort.txt

tableSeparator="<TD align=center>"
echo "<TABLE  BORDER=1 ALIGN=CENTER> " > RateFull.html
echo -e "<TD> $tableSeparator Run $tableSeparator StartTime $tableSeparator Partition $tableSeparator RawEvents $tableSeparator RecoEvents $tableSeparator RecoEvents FNAL $tableSeparator datasets <TR> " >> RateFull.html

#echo -e "\tRun|\tStartTime|\t\t\tPartition|\tRawEvents|\tRecoEvents|\t RecoEvents FNAL|\t datasets" > RateFull.txt
echo -e "\tRun|\tStartTime|\t\t\tPartition|\t\tRawEvents|\tRecoEvents|\t RecoEvents FNAL" > RateShort.txt

#cat physicsRuns.txt | awk -F"\t" '{print $1"\t|\t"$4"\t|\t"$6}'
for run in `cat $physicsRunsFile | awk -F"\t" '{print $1}'`
  do 
  time=`grep "^$run" $physicsRunsFile | awk -F"\t" '{print $4}'`
  partition=`grep "^$run" $physicsRunsFile | awk -F"\t" '{print $2}' | sed -e "s# ##"`
  rawEvents=`grep "^\(\|*\)$run|" $rawdatasetStatFile `
  recoEvents=`grep "^\(\|*\)$run|" $recodatasetStatFile `
  recoEventsFNAL=`grep "^\(\|*\)$run|" $recodatasetStatFileFNAL `
  problem=""
  [ `echo $recoEvents $recoEventsFNAL $rawEvents | grep -c "\*"` -gt 0 ] && problem="o"
  [ "$recoEvents" == "" ] && recoEvents=" |  | "
  [ "$recoEventsFNAL" == "" ] && recoEventsFNAL=" |  | "

#  [ "$run" != "" ] && echo -e "$problem|$run| $time| $partition| $rawEvents| $recoEvents| $recoEventsFNAL" | awk -F "|" '{printf("%1s|%9d| %s|%30s|\t%08d|\t%08d|\t%08d| %50s| %50s| %50s\n",$1,$2,$3,$4,$6,$9,$12,$7,$10,$13)}' >> RateFull.txt
 [ "$run" != "" ] && echo -e "$problem|$run| $time| $partition| $rawEvents| $recoEvents| $recoEventsFNAL" | awk -F "|" '{printf("<TD>%1s|%9d| %s|%30s|\t%08d|\t%08d|\t%08d| %50s| %50s| %50s<TR>\n",$1,$2,$3,$4,$6,$9,$12,$7,$10,$13)}' | sed -e "s@|@<TD align=center>@g" >> RateFull.html

  [ "$run" != "" ] && echo -e "$problem|$run| $time| $partition| $rawEvents| $recoEvents| $recoEventsFNAL" | awk -F "|" '{printf("%1s|%9d| %s|%30s|\t%08d|\t%08d|\t%08d\n",$1,$2,$3,$4,$6,$9,$12)}' >> RateShort.txt

done

cp RateShort.txt RateFull.html /data1/CrabAnalysis/Rate/
}

function getDBSStatistics(){

physicsRunsFile=$1
datasetNameFile=$2
outfile=$3

rm $outfile
for run in `cat $physicsRunsFile | awk -F"\t" '{print $1}'`
  do 
  
  Ndataset=`grep "00$run" $datasetNameFile |  wc -l`
  [ $Ndataset -eq 0 ] && continue
  if [ $Ndataset -gt 1 ]; then
      echo "PROBLEM " && grep "00$run" $datasetNameFile 
      
      stableVal=""
      stableEntr=0
      for datasetName in `grep "00$run" $datasetNameFile | awk -F"'" '{print $4}'`
	do
	value=`python dbsreadprocdatasetStatistics.py --DBSAddress=MCGlobal/Writer --datasetName=$datasetName`
	entr=`echo $value | awk -F"|" '{print $1}'`
	if [ $stableEntr -lt $entr ];
	    then
	    stableEntr=$entr
	    stableVal=$value
	fi
      done
      echo -e "*$run|\t$stableVal|\t$datasetName" >> $outfile
  else
      datasetName=`grep "00$run" $datasetNameFile | awk -F"'" '{print $4}'`
      echo -e "$run|\t`python dbsreadprocdatasetStatistics.py --DBSAddress=MCGlobal/Writer --datasetName=$datasetName`|\t$datasetName" >> $outfile
  fi
done

}

######
# MAIN
######


export LOCALHOME=/analysis/sw/CRAB
## Where to copy all the results                                                                                                                                               
export python_path=/analysis/sw/CRAB

export MYHOME=/analysis/sw/CRAB

export PYTHONPATH=$PYTHONPATH:${python_path}/COMP/DBS/Clients/PythonAPI
export PYTHONPATH=$PYTHONPATH:${python_path}/COMP/DLS/Client/LFCClient
export PYTHONPATH=$PYTHONPATH:${python_path}/COMP/DLS/Client/DliClient
export PATH=$PATH:${python_path}/COMP/:${python_path}/COMP/DLS/Client/LFCClient

[ -e lock_dbs ] && echo "lock_dbs file exists" && exit
touch lock_dbs

lastRun=0
if [ -e lastRun.txt ]; then
    lastRun=`tail -1 lastRun.txt`
else
    rm -v physicsRuns.txt
fi

echo "...running query to RunSummaryTIF"
let lastRun++
GetPhysicsRuns $lastRun
lastRun=`cat physicsRuns.txt | tail -1 | awk '{print $1}'`
echo $lastRun > lastRun.txt

############
### RAW Data
###########
echo "...running python to query DBS.. and take datasets"
python dbsreadprocdatasetList.py --DBSAddress=MCGlobal/Writer --datasetPath=/TAC*DAQ-EDM/RAW/* --logfile=RawDatasetList.txt 
echo "...running python to query DBS.. and take stats"
getDBSStatistics physicsRuns.txt RawDatasetList.txt RawDatasetStat.txt


############
### RECO Data
###########

##########
## BARI
##########

echo "...running python to query DBS.. and take datasets"
python dbsreadprocdatasetList.py --DBSAddress=MCGlobal/Writer --datasetPath=/TAC*DAQ-EDM/RECO/* --logfile=RecoDatasetList.txt 
echo "...running python to query DBS.. and take stats"
getDBSStatistics physicsRuns.txt RecoDatasetList.txt RecoDatasetStat.txt


##########
## FNAL
##########

echo "...running python to query DBS.. and take datasets"
python dbsreadprocdatasetList.py --DBSAddress=MCGlobal/Writer --datasetPath=/TAC*Pass2/RECO/* --logfile=RecoFNALDatasetList.txt 
echo "...running python to query DBS.. and take stats"
getDBSStatistics physicsRuns.txt RecoFNALDatasetList.txt RecoFNALDatasetStat.txt


#########
## Stats
########
echo "...compose table"
GetListOfEvents  physicsRuns.txt RawDatasetStat.txt RecoDatasetStat.txt RecoFNALDatasetStat.txt

rm lock_dbs