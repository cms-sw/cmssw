#!/bin/tcsh

eval `scram runtime -csh`

echo
date +%F\ %a\ %T
echo
echo "Existing cfg files:"
ls -l OnLine*.py

echo
echo "Creating offline cfg files with cmsDriver"
echo "./cmsDriver.csh "$1
time  ./cmsDriver.csh $1

echo
date +%F\ %a\ %T
echo
echo "Running selected cfg files from:"
pwd

rm -f                           ./runOne.log 
time ./runOne.csh DATA    $1 >& ./runOne.log &
time ./runOne.csh MC      $1

  set N = 0
  cp -f ./runOne.log ./runOne.tmp  
  grep -q Finished   ./runOne.tmp
  set F = $?

while ( $F )
  awk "{if (NR>$N) {print}}"  ./runOne.tmp
  set N = `cat ./runOne.tmp | wc -l`
  sleep 13
  cp -f ./runOne.log ./runOne.tmp  
  grep -q Finished   ./runOne.tmp
  set F = $?
end

wait

  awk "{if (NR>$N) {print}}"  ./runOne.log
  rm -f ./runOne.{log,tmp}

echo
echo "Resulting log files:"
ls -l *.log
echo
date +%F\ %a\ %T
