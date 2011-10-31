#!/bin/bash

#set -o verbose

export name=${1} #Put the name corresponding to the needed gridpack in the official repository (without _grid.tar.gz)
echo "running the gridpack $name"

export nevt=${2}
echo "number of events requested = $nevt"

export rnum=${3}
echo "random seed used for the run= $rnum"

# retrieve the wanted gridpack from the official repository 

wget --no-check-certificate http://cms-project-generators.web.cern.ch/cms-project-generators/slc5_ia32_gcc434/madgraph/V5_1.1/7TeV_Summer11/Gridpacks/${name}_gridpack.tar.gz 

# force the f77 compiler to be the CMS defined one

ln -s `which gfortran` f77
ln -s `which gfortran` g77
export PATH=`pwd`:${PATH}

tar xzf ${name}_gridpack.tar.gz ; rm -f ${name}_gridpack.tar.gz ; cd madevent

# run the production stage
./bin/compile
./bin/clean4grid
cd ..
./run.sh ${nevt} ${rnum}

gzip -d events.lhe

exit 0
