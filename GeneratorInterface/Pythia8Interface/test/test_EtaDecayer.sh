#!/bin/bash
# Runs etaDecayer_cfg.py forcing every eta, then one per event, and checks that eta -> mu mu e e decays appear.
set -o pipefail
function die { echo "$1: status $2"; exit $2; }
for onePerEvent in False True; do
  log=etaDecayer_onePerEvent${onePerEvent}.log
  cmsRun ${SCRAM_TEST_PATH}/etaDecayer_cfg.py onePerEvent=${onePerEvent} 2>&1 | tee ${log} || die "cmsRun etaDecayer_cfg.py onePerEvent=${onePerEvent}" $?
  grep -Eq "TrigReport Events total = 10 passed = [1-9][0-9]* " ${log} || die "no eta -> mu mu e e decay found (onePerEvent=${onePerEvent})" 1
done
