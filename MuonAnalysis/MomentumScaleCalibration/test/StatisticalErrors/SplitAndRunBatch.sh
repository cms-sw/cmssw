#!/bin/bash

if [ $# -lt 3 ]
then
     echo "Error, usage is:"
     echo "SplitAndRun.sh N K Q (WD)"
     echo "Where"
     echo "  - N  is the total number of events"
     echo "  - K  is the number of events per job"
     echo "  - Q  is the name of the queue where the jobs will be submitted"
     echo "  - WD is the dir where \"cmsenv\" is done"
     echo "       (optional, default is \"pwd\")"
     exit
fi

N=$1
K=$2
Q=$3
WDir=`pwd`
if [ $# -eq 4 ]
then
    WDir=$4
fi

let "NJOBS=$N/$K"
if [ $NJOBS -eq 0 ]
then
    echo "Error: N($N) < K($K). Total events must be >= number of events per job"
fi

DirName="`pwd`/StatErrors_`\date +%y%m%d"_"%H%M%S`"
if [ ! -d $DirName ]
then
    mkdir $DirName
fi

echo "Creating and submitting $NJOBS new jobs to queue $Q"

for i in `seq $NJOBS`
do
    echo "Creating and submitting job $i..."
    let "skip=$K*($i-1)"
    #mkdir -p Job_$i
    cat TreeSplitter_cfg.py | sed s/SUBSAMPLEFIRSTEVENT/${skip}/g | sed s/SUBSAMPLEMAXEVENTS/$K/g | sed s/MAXEVENTS/$N/g > ${DirName}/TreeSplitter_${i}_cfg.py
    cat singleJob.lsf | sed s?TEMPLATECMSDIR?${WDir}?g | sed s?TEMPLATEOUTDIR?${DirName}?g | sed s?JOBINDEX?${i}?g > ${DirName}/singleJob_${i}.lsf
    cd ${DirName}
    chmod +x singleJob_${i}.lsf
    bsub -o tmp_${i}.log -q ${Q} singleJob_${i}.lsf
    cd -
done
