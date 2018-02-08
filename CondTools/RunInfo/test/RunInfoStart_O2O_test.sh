#!/bin/sh
conddb --yes copy runinfo_start_31X_hlt --destdb runinfo_O2O_test.db --o2oTest
lastRun=`conddb listRuns | tail -2 | awk '{print $1}'`
echo "last run = $lastRun"
echo "TNS_ADMIN: $TNS_ADMIN"
cat /etc/tnsnames.ora | grep cms_orcon_adg
cmsRun ./src/CondTools/RunInfo/python/RunInfoPopConAnalyzer.py runNumber=$lastRun destinationConnection=sqlite_file:runinfo_O2O_test.db tag=runinfo_start_31X_hlt
conddb --db runinfo_O2O_test.db list runinfo_start_31X_hlt