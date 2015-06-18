#!/bin/tcsh

cmsenv
rehash

echo
date +%F\ %a\ %T
echo
echo "Existing cfg files:"
ls -l OnLine*.py

#echo
#echo "Creating OnLine cfg files adding the HLTAnalyzerEndpath:"
#
#foreach gtag ( Data Mc )
#  set GTAG = ` echo $gtag | tr "[a-z]" "[A-Z]" `
#  foreach table ( FULL GRun 50nsGRun LowPU HIon PIon 25ns14e33_v1 50ns_5e33_v1 Fake )
#    set oldfile = OnLine_HLT_${table}.py
#    set newfile = OnLine_HLT_${table}_${GTAG}.py
#    ln -s $oldfile $newfile
#  end
#end
#
#echo
#echo "Created OnLine cfg files:"
#ls -l OnLine*.py

echo
echo "Creating offline cfg files with cmsDriver"
echo "./cmsDriver.csh"
time  ./cmsDriver.csh

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
