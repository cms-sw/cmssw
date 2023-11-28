#!/bin/sh
conddb --yes copy runinfo_start_31X_hlt --destdb runinfo_O2O_test.db --o2oTest
lastRun=`conddb listRuns | tail -2 | awk '{print $1}'`
echo "last run = $lastRun"
cmsRun $CMSSW_BASE/src/CondTools/RunInfo/python/RunInfoPopConAnalyzer.py runNumber=$lastRun destinationConnection=sqlite_file:runinfo_O2O_test.db tag=runinfo_start_31X_hlt
ret=$?
conddb --db runinfo_O2O_test.db list runinfo_start_31X_hlt
echo "return code is $ret"
exit $ret
