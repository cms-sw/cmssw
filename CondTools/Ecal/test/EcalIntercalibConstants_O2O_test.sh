#!/bin/sh
conddb --yes copy EcalIntercalibConstants_V1_hlt --destdb EcalIntercalibConstants_V1_hlt_O2OTEST.db --o2oTest
conddb --yes copy EcalIntercalibConstants_0T --destdb EcalIntercalibConstants_V1_hlt_O2OTEST.db
conddb --yes copy EcalIntercalibConstants_3.8T_v2 --destdb EcalIntercalibConstants_V1_hlt_O2OTEST.db
lastRun=`conddb --db EcalIntercalibConstants_V1_hlt_O2OTEST.db list EcalIntercalibConstants_V1_hlt  | tail -2 | head -1 | awk '{print $1}'`
conddb --yes copy runinfo_start_31X_hlt --destdb EcalIntercalibConstants_V1_hlt_O2OTEST.db -f $lastRun -t $lastRun
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/EcalIntercalibConstantsPopConBTransitionAnalyzer_cfg.py runNumber=$lastRun destinationDatabase=sqlite_file:EcalIntercalibConstants_V1_hlt_O2OTEST.db destinationTag=EcalIntercalibConstants_V1_hlt tagForRunInfo=runinfo_start_31X_hlt tagForBOff=EcalIntercalibConstants_0T tagForBOn=EcalIntercalibConstants_3.8T_v2
ret=$?
conddb --db EcalIntercalibConstants_V1_hlt_O2OTEST.db list EcalIntercalibConstants_V1_hlt
echo "return code is $ret"
exit $ret
