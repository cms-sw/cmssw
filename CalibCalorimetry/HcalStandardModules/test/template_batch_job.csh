#!/bin/csh
# current Dir, pwd
cd <dir>
cmsenv
cmsRun <job>
rfcp <run>-peds_ADC_*.txt <castor>
zip pedstxt.zip *-peds_ADC*.txt
rm *-peds_ADC*
rm <job>
