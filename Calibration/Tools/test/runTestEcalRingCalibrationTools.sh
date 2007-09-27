#!/bin/sh
cmsRun runTestEcalRingCalibrationTools.cfg
head -n 361 eb.ringDump | enscript -r -f Courier3 -o eb_minus.ringDump.ps -L340 
head -n 732 eb.ringDump | tail -n 361 | enscript -r -f Courier3 -o eb_plus.ringDump.ps -L340 
head -n 101 ee.ringDump | enscript -r -f Courier3 -o ee_minus.ringDump.ps -L400
head -n 212 ee.ringDump | tail -n 101 | enscript -r -f Courier3 -o ee_plus.ringDump.ps -L400
