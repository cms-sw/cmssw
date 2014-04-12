#!/bin/bash

#set -o verbose

echo "   ______________________________________     "
echo "         Running Madgraph5                    "
echo "   ______________________________________     "

repo=${1}
echo "%MSG-MG5 repository = $repo"

name=${2} 
echo "%MSG-MG5 gridpack = $name"

nevt=${3}
echo "%MSG-MG5 number of events requested = $nevt"

rnum=${4}
echo "%MSG-MG5 random seed used for the run = $rnum"

LHEWORKDIR=`pwd`

if [[ -d madevent ]]
    then
    echo 'madevent directory found'
    echo 'Setting up the environment'
    rm -rf madevent
fi
mkdir madevent; cd madevent

# retrieve the wanted gridpack from the official repository 
fn-fileget -c `cmsGetFnConnect frontier://smallfiles` ${repo}/${name}_tarball.tar.gz 
#cp -p /afs/cern.ch/work/b/bendavid/CMSSWgen62/genproductions/bin/aMCatNLO/${name}_tarball.tar.gz ./

#check the structure of the tarball
tar xzf ${name}_tarball.tar.gz ; rm -f ${name}_tarball.tar.gz ;

# force the f77 compiler to be the CMS defined one
ln -sf `which gfortran` f77
ln -sf `which gfortran` g77
PATH=`pwd`:${PATH}

cd mgbasedir/${name}

#replace the seed in the run card with ${rnum}
sed -i "s#[0-9]\+ *= *iseed# ${rnum} = iseed#g" Cards/run_card.dat

#replace the number of events in the run_card
sed -i "s#[0-9]\+ *= *nevents# ${nevt} = nevents#g" Cards/run_card.dat

run_card_seed=`fgrep "${rnum} = iseed" Cards/run_card.dat | awk '{print $1}'`
run_card_nevents=`fgrep "${nevt} = nevents" Cards/run_card.dat | awk '{print $1}'`
  
if [[ $run_card_seed -eq $rnum ]] ;then
  echo "run_card_seed = ${run_card_seed}"
else 
  echo "%MSG-MG5 Error: Seed numbers $run_card_seed doesnt match ( $rnum )"
  exit 1
fi

if [[ $run_card_nevents -eq $nevt ]] ;then
  echo "run_card_nevents = ${run_card_nevents}"
else
  echo "%MSG-MG5 Error: Number of events $run_card_nevents doesnt match ( $nevt )"
  exit 1
fi

if [ -f Cards/madspin_card.dat ] ;then
  #set random seed for madspin
  rnum2=$(($rnum+1000000))
  sed -i "s#\# set seed 1# set seed ${rnum2}#g" Cards/madspin_card.dat
  madspin_card_seed=`fgrep " set seed ${rnum2}" Cards/madspin_card.dat | awk '{print $3}'`
  if [[ $madspin_card_seed -eq $rnum2 ]] ;then
    echo "madevent seed = $rnum, madspin seed = $rnum2, madspin_card_seed = $madspin_card_seed"
  else
    echo "%MSG-MG5 Error: Madspin seed $madspin_card_seed doesnt match ( $rnum2 )"
    exit 1
  fi
fi

#generate events
bin/generate_events -fox -n ${name}

if [[ -d Events/${name}_decayed_1 ]]
    then 
    mv Events/${name}_decayed_1/events.lhe.gz $LHEWORKDIR/${name}_final.lhe.gz
else
    mv Events/${name}/events.lhe.gz $LHEWORKDIR/${name}_final.lhe.gz
fi

cd $LHEWORKDIR
gzip -d ${name}_final.lhe.gz
#cp ${name}_final.lhe ${name}_final.lhe.bak

ls -l
echo

exit 0

