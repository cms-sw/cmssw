#!/bin/bash

#script to run generic lhe generation tarballs
#kept as simply as possible to minimize need
#to update the cmssw release
#(all the logic goes in the run script inside the tarball
# on frontier)
#J.Bendavid

echo "   ______________________________________     "
echo "         Running Generic Tarball/Gridpack     "
echo "   ______________________________________     "

repo=${1}
echo "%MSG-MG5 repository = $repo"

name=${2} 
echo "%MSG-MG5 gridpack = $name"

nevt=${3}
echo "%MSG-MG5 number of events requested = $nevt"

rnum=${4}
echo "%MSG-MG5 random seed used for the run = $rnum"

ncpu=${5}
echo "%MSG-MG5 thread count requested = $ncpu"

LHEWORKDIR=`pwd`

if [[ -d lheevent ]]
    then
    echo 'lheevent directory found'
    echo 'Setting up the environment'
    rm -rf lheevent
fi
mkdir lheevent; cd lheevent

# retrieve the wanted gridpack from the official repository 
fn-fileget -c `cmsGetFnConnect frontier://smallfiles` ${repo}/${name}

#check the structure of the tarball
tar xaf ${name} ; rm -f ${name} ;

#generate events
./runcmsgrid.sh $nevt $rnum $ncpu

mv cmsgrid_final.lhe $LHEWORKDIR/

cd $LHEWORKDIR

#cleanup working directory (save space on worker node for edm output)
rm -rf lheevent

exit 0

