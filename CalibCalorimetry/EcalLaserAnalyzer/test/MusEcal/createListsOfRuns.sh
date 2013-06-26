#!/bin/bash

prim() {

usage="Usage: ./prim [OPTION] <iz=0,1,2,3> <iSM=1,...18>"
[ -z $1 ] && [ -z $2 ] && echo $usage && return

dir="EB"
[ $1 -eq 1 ] || [ $1 -eq 2 ] || dir="EE"

iz=$1
ism=$2

topdir=${MESTORE}
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

echo '> Get the lists of runs (Blue Laser, Red Laser, Test Pulse) for ' $dir
$MUSECAL/getListOfRuns.pl $datadir $dir 

echo '> Done.'

return
}

echo 'Preparing list of runs for ECAL laser primitives'
for i in 1 2
do
  for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
  do
    prim $i $j
  done
done
for j in 1 2 3 4 5 6 7 8 9
do
  prim 0 $j
done
for j in 1 2 3 4 5 6 7 8 9
do
  prim 3 $j
done
