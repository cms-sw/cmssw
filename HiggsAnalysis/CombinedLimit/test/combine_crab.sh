#!/bin/sh
#########################
#
# Driver script for Toy Monte Carlo submission with CRAB 
#
# author: Luca Lista, INFN
#                      
#########################

if [ -e outputToy ]; then 
  rm -rf outputToy 
fi
mkdir outputToy

i="$1"
if [ "$i" == "help" ]; then
  echo "usage: combine_crab.sh <job index> [<max events>]"
  exit 0;
fi
if [ "$i" = "" ]; then
  echo "Error: missing job index"
  exit 1;
fi
echo "max events from CRAB: $MaxEvents"
n="$MaxEvents" 
if [ "$n" = "" ]; then
  n="$2"
fi
if [ "$n" = "" ]; then
  echo "Error: missing number of experiments"
  exit 2;
fi

# first, link locally the executable:
# ln -s ../../../../bin/slc5_amd64_gcc434/combine .

echo "job number: seed #$i with $n toys"
./combine model.root -t$n -s$i >& log.txt
mv *.root outputToy/
mv log.txt outputToy/
echo "pack the results"
tar cvfz outputToy.tgz outputToy/

