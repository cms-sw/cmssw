#!/bin/bash
set prefix1="/tmp/BeamSpotDipServer"
set prefix2="/tmp/BeamPixelDipServer"
set suffix=`date '+%G%m%d%H%M'`
set fileName1=${prefix1}_${suffix}.log
set fileName2=${prefix2}_${suffix}.log
echo $fileName1
echo $fileName2

## Test
java cms.dip.tracker.beamspot.BeamSpotDipServer false false dip/CMS/Tracker/BeamSpotTest dip/CMS/LHCTEST/LuminousRegion_BeamSpot /nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults.txt 5 10 /nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults_TkStatus.txt > ${fileName1} &
sleep 5
java cms.dip.tracker.beamspot.BeamSpotDipServer false false dip/CMS/Tracker/BeamPixelTest dip/CMS/LHCTEST/LuminousRegion_BeamPixel /nfshome0/dqmpro/BeamMonitorDQM/BeamPixelResults.txt 5 10 /nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults_TkStatus.txt > ${fileName2} &

