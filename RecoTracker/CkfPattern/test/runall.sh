#!/bin/bash

#Dummy script to run all tests

testsTracking="GenToSimHits.cfg    
SimHitsToDigis.cfg
DigisToRecHits.cfg          
RecHitsToSeeds.cfg  
SeedToTrackCandidates.cfg  
TrackCandidatesToTracks.cfg
DigisToTrackCandidates.cfg
RecHitsToTracks_PixelOnlySeeded.cfg  
RecHitsToTracks_PixelLessSeeded.cfg  
RecHitsToTracks_MixedSeeded.cfg
regionalTracking.cfg  
"

tests=`echo $testsTracking`

report=""

let nfail=0
let npass=0

echo "Tests to be run : " $tests

eval `scramv1 runtime -sh`

for file in $tests 
do
    echo Preparing to run $file
    cmsRun $file
    if [ $? -ne 0 ] ;then
      echo "cmsRun $file : FAILED"
      report="$report \n cmsRun $file : FAILED"
      let nfail+=1
    else 
      echo "cmsRun $file : PASSED"
      report="$report \n cmsRun $file : PASSED"
      let npass+=1
    fi 
done


report="$report \n \n $npass tests passed, $nfail failed \n"

echo -e "$report" 
#rm -f runall-report.log
#echo -e "$report" >& runall-report.log
