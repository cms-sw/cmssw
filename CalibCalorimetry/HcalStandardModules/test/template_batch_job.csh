#!/bin/csh
# current Dir, pwd
cd <dir>
cmsenv
cmsRun <job>
rfcp <run>-peds_ADC_*.txt /castor/cern.ch/user/a/andrey/peds2011/
zip pedstxt.zip *-peds_ADC*.txt
rm *-peds_ADC*
