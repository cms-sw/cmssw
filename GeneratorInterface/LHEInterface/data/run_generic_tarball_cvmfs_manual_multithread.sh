#!/bin/bash

#script to run generic lhe generation tarballs
#kept as simply as possible to minimize need
#to update the cmssw release
#(all the logic goes in the run script inside the tarball
# on frontier)
#J.Bendavid

#exit on first error
set -e

echo "   ______________________________________     "
echo "         Running Generic Tarball/Gridpack     "
echo "   ______________________________________     "

path=${1}
echo "gridpack tarball path = $path"

nevt=${2}
echo "%MSG-MG5 number of events requested = $nevt"

rnum=${3}
echo "%MSG-MG5 random seed used for the run = $rnum"

ncpu=${4}
echo "%MSG-MG5 thread count requested = $ncpu"

echo "%MSG-MG5 residual/optional arguments = ${@:5}"

if [ -n "${5}" ]; then
  use_gridpack_env=${5}
  echo "%MSG-MG5 use_gridpack_env = $use_gridpack_env"
fi

if [ -n "${6}" ]; then
  scram_arch_version=${6}
  echo "%MSG-MG5 override scram_arch_version = $scram_arch_version"
fi

if [ -n "${7}" ]; then
  cmssw_version=${7}
  echo "%MSG-MG5 override cmssw_version = $cmssw_version"
fi

LHEWORKDIR=`pwd`
echo "%MSG-MG5 LHEWORKDIR = "${LHEWORKDIR}

if [ "$use_gridpack_env" = false -a -n "$scram_arch_version" -a -n  "$cmssw_version" ]; then
  echo "%MSG-MG5 CMSSW version = $cmssw_version"
  export SCRAM_ARCH=${scram_arch_version}
  scramv1 project CMSSW ${cmssw_version}
  cd ${cmssw_version}/src
  eval `scramv1 runtime -sh`
  cd $LHEWORKDIR
fi

##########################
echo "%MSG-MG5 multithread loop start"
##########################
nevt_run=$(echo print $nevt/$ncpu | python)
resid_run=$(echo print $nevt%$ncpu | python)

for (( thread=0; thread<$ncpu; thread++ ))
do

    echo "%MSG-MG5 thread "${thread}" started"
    if [[ -d lheevent_$thread ]]
        then
        echo '%MSG-MG5 lheevent_'$thread' directory found, removing before to proceed'
        rm -rf lheevent_$thread
    fi
    mkdir lheevent_$thread; cd lheevent_$thread

    echo "%MSG-MG5 untar the tarball from cvmfs"
    tar -xaf ${path} 

    if [[ thread -eq 0 ]]; then
        #generate events
        ./runcmsgrid.sh $((nevt_run+resid_run)) $rnum 1 ${@:5} &
    else
        #generate events
        ./runcmsgrid.sh $((nevt_run)) $rnum 1 ${@:5} &
    fi
    
    rnum=$((rnum+10))
    
    cd $LHEWORKDIR

done
##########################
# multithread loop end
##########################

wait # wait for all the subprocesses to finish

for (( thread=0; thread<$ncpu; thread++ ))
do
    mv lheevent_$thread/cmsgrid_final.lhe $LHEWORKDIR/cmsgrid_final_$thread.lhe
done

# merge multiple lhe files if needed
ls -lrt $LHEWORKDIR/cmsgrid_final_*.lhe
if [  $thread -gt "1" ]; then
    echo "%MSG-MG5 Merging files and deleting unmerged ones"
    cp /cvmfs/cms.cern.ch/phys_generator/gridpacks/lhe_merger/merge.pl ./
    chmod 755 merge.pl
    ./merge.pl $LHEWORKDIR/cmsgrid_final_*.lhe cmsgrid_final.lhe.gz banner.txt
    gzip -d cmsgrid_final.lhe.gz
    rm $LHEWORKDIR/cmsgrid_final_*.lhe banner.txt;
else
    mv $LHEWORKDIR/cmsgrid_final_$thread.lhe $LHEWORKDIR/cmsgrid_final.lhe
fi


cd $LHEWORKDIR

#cleanup working directory (save space on worker node for edm output)
rm -rf lheevent_*

exit 0

