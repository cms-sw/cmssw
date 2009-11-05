#! /bin/bash

# $1 : config file
# $2 : output directory
# $3 : output DQM directory (optional)

curdir="$(pwd)"
#workdir="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/bonato/DEVEL/HIPWorkflow/CMSSW_3_2_4/src/"
workdir="<MYCMSSW>"
dqmdir="MICKEY"


if [ $# == 3 ]
then 
dqmdir="$3"
else
dqmdir="${curdir}/MONITORING/DQM/"
fi


# set up the CMS environment (choose your release and working area):
echo Setting up CMSSW environment in $workdir
cd $workdir
eval `scramv1 runtime -sh`
export STAGE_SVCCLASS=cmscafuser
###rehash ### useless in bash shell, only for tcsh



echo Running in $curdir...
cd $curdir

rfcp $1 $2/logfiles/
#cp   $1 "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/bonato/DEVEL/HIPWorkflow/ALCARECOskim/v1.3/MONITORING/logfiles/"

BASE_JOBNAME=$(basename "$1" .py)
LOGFILE=$BASE_JOBNAME.log
OUTFILE=$BASE_JOBNAME.out
TRKFILE=$BASE_JOBNAME"_TrackStats.root"
HITFILE=$BASE_JOBNAME"_HitMaps.root"

time cmsRun $1 &> $LOGFILE

echo
echo "---------"
echo "File list in $(pwd): "
ls -lh
echo "---------"

export STAGE_SVCCLASS=cmscafuser
#rfcp *Skimmed*.root "$2"

for dqmfile in $(ls  *TracksStatistics*.root)
do
#dqmbase=$(basename "$dqmfile" .root)
if [[ "$dqmfile" =~ "CTF" ]]; then rfcp $dqmfile $dqmdir/CTF/$TRKFILE ; fi
if [[ "$dqmfile" =~ "CosmicTF" ]]; then rfcp $dqmfile $dqmdir/CosmicTF/$TRKFILE ; fi
done

for dqmfile in $(ls  *HitMaps*.root)
do
#dqmbase=$(basename "$dqmfile" .root)
#rfcp $dqmfile $dqmdir/$dqmbase"_HitMaps.root"

if [[ "$dqmfile" =~ "CTF" ]]; then rfcp $dqmfile $dqmdir/CTF/$HITFILE ; fi
if [[ "$dqmfile" =~ "CosmicTF" ]]; then rfcp $dqmfile $dqmdir/CosmicTF/$HITFILE ; fi
done

rm -f *TracksStatistics*.root  *HitMaps*.root
rfcp ALCASkim*.root "$2/"
rfcp *.log "$2/logfiles/"
rfcp *.out "$2/logfiles/"



#### # for skimfile in $( ls  ALCASkim*.root )
#### # do
#### # nsrename $skimfile $2/$skimfile
#### # stager_get -S 'cmscafuser' -M $2/$skimfile
#### # done


rfcp *.log "${dqmdir}/../logfiles/"
rfcp *.out "${dqmdir}/../$OUTFILE"


### Clean up
rm -f  *.root
rm -f *.log
rm -f *.out

for logfile in $( ls ${dqmdir}/../logfiles/*.log ) 
do
gzip  $logfile
##rm -f $logfile
done 
