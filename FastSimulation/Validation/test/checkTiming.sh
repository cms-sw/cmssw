#!/bin/sh

# First, enable timing report and message logger
cp $CMSSW_RELEASE_BASE/src/FastSimulation/Configuration/test/IntegrationTestWithHLT_cfg.py IntegrationTestWithHLTWithTiming_cfg.py
echo 'process.Timing =  cms.Service("Timing")'  >> IntegrationTestWithHLTWithTiming_cfg.py
echo 'process.load("FWCore/MessageService/MessageLogger_cfi")' >> IntegrationTestWithHLTWithTiming_cfg.py
echo 'process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt")' >> IntegrationTestWithHLTWithTiming_cfg.py 

#run and measure total fime
/usr/bin/time -o timefrac -f'%P' oval run cmsRun.runWithTiming
rm IntegrationTestWithHLTWithTiming_cfg.py

# extract timing info
grep "TimeModule>" pyDetailedInfo.txt > TimingInfo.txt

# compile the executable
timefraction=`sed -e 's/\%//' timefrac`
$CMSSW_BASE/test/slc4_ia32_gcc345/timing -t $timefraction -n TimingInfo.txt -o > tmp_file
grep -A 9999 "Timing per module" tmp_file | grep -B 9999 "Timing per label" | grep OVAL | sort -k 3

