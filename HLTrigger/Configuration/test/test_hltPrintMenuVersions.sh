#!/bin/bash

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  if [ $# -gt 2 ]; then
    printf "%s\n" "=== Log File =========="
    cat $3
    printf "%s\n" "=== End of Log File ==="
  fi
  exit $2
}

# run test job
test_hltPrintMenuVersions_1="hltPrintMenuVersions /dev/CMSSW_13_0_0/Fake2"

${test_hltPrintMenuVersions_1} &> test_hltPrintMenuVersions_log \
  || die "Failure '${test_hltPrintMenuVersions_1}'" $? test_hltPrintMenuVersions_log

cat <<@EOF > test_hltPrintMenuVersions_log_expected
HLT Configuration: /dev/CMSSW_13_0_0/Fake2 (database = "run3")

   * =/dev/CMSSW_13_0_0/Fake2/V9 (CMSSW_13_0_2)=: !Migration to release template of !CMSSW_13_0_2
   * =/dev/CMSSW_13_0_0/Fake2/V8 (CMSSW_13_0_1)=: !Migration
   * =/dev/CMSSW_13_0_0/Fake2/V7 (CMSSW_13_0_0)=: [[https://its.cern.ch/jira/browse/CMSHLT-2651][CMSHLT-2651]]: changed compression settings of all !OutputModules to (algorithm=ZSTD, level=3)
   * =/dev/CMSSW_13_0_0/Fake2/V6 (CMSSW_13_0_0)=: !Migration
   * =/dev/CMSSW_13_0_0/Fake2/V5 (CMSSW_13_0_0_pre4)=: !Migration
   * =/dev/CMSSW_13_0_0/Fake2/V4 (CMSSW_13_0_0_pre3)=: !Migration
   * =/dev/CMSSW_13_0_0/Fake2/V3 (CMSSW_13_0_0_pre2)=: !Migration
   * =/dev/CMSSW_13_0_0/Fake2/V2 (CMSSW_13_0_0_pre1)=: !Migration
   * =/dev/CMSSW_13_0_0/Fake2/V1 (CMSSW_12_6_0_pre5)=: saveAs /dev/CMSSW_12_6_0/Fake2/V6 [7242]

@EOF

diff test_hltPrintMenuVersions_log test_hltPrintMenuVersions_log_expected \
  || die "Unexpected differences in outputs of '${test_hltPrintMenuVersions_1}'" $?
