#!/bin/bash

prim() {

usage="Usage: ./prim [OPTION] <iz=0,1,2,3> <iSM=1,...18>"
[ -z $1 ] && [ -z $2 ] && echo $usage && return

dir="EB"
[ $1 -eq 1 ] || [ $1 -eq 2 ] || dir="EE"

iz=$1
ism=$2

topdir=${MESTORE}
firstrun=${PRIM_FIRST_RUN}
lastrun=${PRIM_LAST_RUN}

echo '> Top directory : ' $topdir
if [ $iz -lt 2 ]
then
    dir=${dir}-${ism}
else
    dir=${dir}+${ism}
fi

datapath=$MELMDAT
datadir=${datapath}/${dir}
if [ -d $datadir ] 
then
    echo '> Data directory exists: ' $datadir
else
    echo '> Data directory does not exist: ' $datadir '. Return'
    return
fi

if [ -d ${topdir} ] 
then
    echo '> Store directory exists: ' $dir
else
    echo '> Check Top Directory first. Exit.'
    return
fi

dir=${topdir}/${dir}
if [ -d ${dir} ] 
then
    echo '> Store directory exists: ' $dir
else
    echo '> Creating directory ' $dir
    mkdir $dir
fi

echo '> Get the lists of runs (Blue Laser, Red Laser, Test Pulse, LED) for ' $dir
$MUSECAL/getListOfRuns.pl $datadir $dir $firstrun $lastrun

echo '> Done.'

return
}

echo 'Preparing list of runs for ECAL laser primitives'
if [ $1 = 0 ]; then        
    testifyfile=$MELPRIM/creatingListsOfRuns_EBMinus
fi
if [ $1 = 1 ]; then        
    testifyfile=$MELPRIM/creatingListsOfRuns_EBPlus
fi
if [ $1 = 2 ]; then        
    testifyfile=$MELPRIM/creatingListsOfRuns_EE
fi

echo $testifyfile
touch $testifyfile

if [ -e $testifyfile ] 
then
    echo ' File created: ' $testifyfile 
else
    echo ' Error: '$testifyfile ' not created '
fi

iEBChoice=0
iEEChoice=0

if [ $1 = 0 ]; then        
    iEBChoice=1
fi
if [ $1 = 1 ]; then        
    iEBChoice=2
fi

if [ $1 = 2 ]; then        
    iEEChoice=1
fi



for i in 1 2
do
  for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
  do
    if [ $i = $iEBChoice ]; then    
	prim $i $j
	echo ' Doing prim ' $i $j
    fi
  done
done

if [ $iEEChoice = 1 ]; then 
    for j in 1 2 3 4 5 6 7 8 9
      do
      echo ' Doing prim 0 ' $j
      prim 0 $j
    done
    for j in 1 2 3 4 5 6 7 8 9
      do
      echo ' Doing prim 3 ' $j
      prim 3 $j
    done
fi


rm -f $testifyfile

if [ -e $testifyfile ] 
then
    echo ' Error: file not erased: ' $testifyfile 
else
    echo ' File erased: '$testifyfile 
fi