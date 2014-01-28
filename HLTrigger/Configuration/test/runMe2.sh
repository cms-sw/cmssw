#! /bin/bash

echo "Creating offline configs with cmsDriver"
echo "./cmsDriver.sh"
      ./cmsDriver.sh

for lumi in GRun HIon; do
  for task in OnLine_HLT OnData_HLT; do
    echo
    name=${task}_${lumi}
    rm -f $name.{log,root}
    echo "cmsRun $name.py >& $name.log"
          cmsRun $name.py >& $name.log
  done &        # run the different lumi in parallel
done

wait            # wait for all process to complete

for lumi in 8E29 1E31 HIon; do
  for task in RelVal_DigiL1Raw RelVal_HLT RelVal_HLT2 RelVal_Reco; do
    echo
    name=${task}_${lumi}
    rm -f $name.{log,root}
    echo "cmsRun $name.py >& $name.log"
          cmsRun $name.py >& $name.log
  done &        # run the different lumi in parallel
done

wait            # wait for all process to complete
