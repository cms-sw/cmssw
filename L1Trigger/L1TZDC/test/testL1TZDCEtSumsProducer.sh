#!/bin/bash

# Pass in name and status
function die {
  echo $1: status $2
  exit $2
}

# run test job
TESTDIR="${SCRAM_TEST_PATH}"

set -o pipefail

cmsRun "${TESTDIR}"/testL1TZDCEtSumsProducer_cfg.py -n 10 2>&1 | tee log_testL1TZDCEtSumsProducer \
 || die "Failure running testL1TZDCEtSumsProducer_cfg.py" $?

set +o pipefail

grep l1tZDCEtSumsPrinter2 log_testL1TZDCEtSumsProducer > log_testL1TZDCEtSumsProducer_l1tZDCEtSumsPrinter2

# expected PathSummary of test job
cat <<@EOF > log_testL1TZDCEtSumsProducer_l1tZDCEtSumsPrinter2_expected
[l1tZDCEtSumsPrinter2] etSums[-1][0] (type, hwPt) = (27, 2)
[l1tZDCEtSumsPrinter2] etSums[0][0] (type, hwPt) = (27, 3)
[l1tZDCEtSumsPrinter2] etSums[1][0] (type, hwPt) = (27, 4)
[l1tZDCEtSumsPrinter2] etSums[2][0] (type, hwPt) = (27, 6)
[l1tZDCEtSumsPrinter2] etSums[-1][0] (type, hwPt) = (28, 7)
[l1tZDCEtSumsPrinter2] etSums[-1][1] (type, hwPt) = (27, 2)
[l1tZDCEtSumsPrinter2] etSums[0][0] (type, hwPt) = (28, 12)
[l1tZDCEtSumsPrinter2] etSums[0][1] (type, hwPt) = (27, 1)
[l1tZDCEtSumsPrinter2] etSums[1][0] (type, hwPt) = (28, 0)
[l1tZDCEtSumsPrinter2] etSums[1][1] (type, hwPt) = (27, 1)
[l1tZDCEtSumsPrinter2] etSums[2][0] (type, hwPt) = (28, 0)
[l1tZDCEtSumsPrinter2] etSums[2][1] (type, hwPt) = (27, 2)
[l1tZDCEtSumsPrinter2] etSums[-1][0] (type, hwPt) = (28, 0)
[l1tZDCEtSumsPrinter2] etSums[-1][1] (type, hwPt) = (27, 3)
[l1tZDCEtSumsPrinter2] etSums[0][0] (type, hwPt) = (28, 1023)
[l1tZDCEtSumsPrinter2] etSums[0][1] (type, hwPt) = (27, 6)
[l1tZDCEtSumsPrinter2] etSums[1][0] (type, hwPt) = (28, 0)
[l1tZDCEtSumsPrinter2] etSums[1][1] (type, hwPt) = (27, 5)
[l1tZDCEtSumsPrinter2] etSums[2][0] (type, hwPt) = (28, 1)
[l1tZDCEtSumsPrinter2] etSums[2][1] (type, hwPt) = (27, 7)
[l1tZDCEtSumsPrinter2] etSums[-1][0] (type, hwPt) = (28, 3)
[l1tZDCEtSumsPrinter2] etSums[-1][1] (type, hwPt) = (27, 9)
[l1tZDCEtSumsPrinter2] etSums[0][0] (type, hwPt) = (28, 31)
[l1tZDCEtSumsPrinter2] etSums[0][1] (type, hwPt) = (27, 12)
[l1tZDCEtSumsPrinter2] etSums[1][0] (type, hwPt) = (28, 31)
[l1tZDCEtSumsPrinter2] etSums[1][1] (type, hwPt) = (27, 8)
[l1tZDCEtSumsPrinter2] etSums[2][0] (type, hwPt) = (28, 31)
[l1tZDCEtSumsPrinter2] etSums[2][1] (type, hwPt) = (27, 215)
[l1tZDCEtSumsPrinter2] etSums[-1][0] (type, hwPt) = (28, 0)
[l1tZDCEtSumsPrinter2] etSums[-1][1] (type, hwPt) = (27, 6)
[l1tZDCEtSumsPrinter2] etSums[0][0] (type, hwPt) = (28, 8)
[l1tZDCEtSumsPrinter2] etSums[0][1] (type, hwPt) = (27, 5)
[l1tZDCEtSumsPrinter2] etSums[1][0] (type, hwPt) = (28, 14)
[l1tZDCEtSumsPrinter2] etSums[1][1] (type, hwPt) = (27, 4)
[l1tZDCEtSumsPrinter2] etSums[2][0] (type, hwPt) = (28, 16)
[l1tZDCEtSumsPrinter2] etSums[2][1] (type, hwPt) = (27, 2)
[l1tZDCEtSumsPrinter2] etSums[-1][0] (type, hwPt) = (28, 8)
[l1tZDCEtSumsPrinter2] etSums[-1][1] (type, hwPt) = (27, 4)
[l1tZDCEtSumsPrinter2] etSums[0][0] (type, hwPt) = (28, 5)
[l1tZDCEtSumsPrinter2] etSums[0][1] (type, hwPt) = (27, 1023)
[l1tZDCEtSumsPrinter2] etSums[1][0] (type, hwPt) = (28, 6)
[l1tZDCEtSumsPrinter2] etSums[1][1] (type, hwPt) = (27, 0)
[l1tZDCEtSumsPrinter2] etSums[2][0] (type, hwPt) = (28, 6)
[l1tZDCEtSumsPrinter2] etSums[2][1] (type, hwPt) = (27, 6)
[l1tZDCEtSumsPrinter2] etSums[-1][0] (type, hwPt) = (28, 1)
[l1tZDCEtSumsPrinter2] etSums[-1][1] (type, hwPt) = (27, 61)
[l1tZDCEtSumsPrinter2] etSums[0][0] (type, hwPt) = (28, 36)
[l1tZDCEtSumsPrinter2] etSums[0][1] (type, hwPt) = (27, 55)
[l1tZDCEtSumsPrinter2] etSums[1][0] (type, hwPt) = (28, 5)
[l1tZDCEtSumsPrinter2] etSums[1][1] (type, hwPt) = (27, 30)
[l1tZDCEtSumsPrinter2] etSums[2][0] (type, hwPt) = (28, 2)
[l1tZDCEtSumsPrinter2] etSums[2][1] (type, hwPt) = (27, 18)
[l1tZDCEtSumsPrinter2] etSums[-1][0] (type, hwPt) = (28, 0)
[l1tZDCEtSumsPrinter2] etSums[-1][1] (type, hwPt) = (27, 0)
[l1tZDCEtSumsPrinter2] etSums[0][0] (type, hwPt) = (28, 131)
[l1tZDCEtSumsPrinter2] etSums[0][1] (type, hwPt) = (27, 121)
[l1tZDCEtSumsPrinter2] etSums[1][0] (type, hwPt) = (28, 9)
[l1tZDCEtSumsPrinter2] etSums[1][1] (type, hwPt) = (27, 22)
[l1tZDCEtSumsPrinter2] etSums[2][0] (type, hwPt) = (28, 27)
[l1tZDCEtSumsPrinter2] etSums[2][1] (type, hwPt) = (27, 31)
[l1tZDCEtSumsPrinter2] etSums[-1][0] (type, hwPt) = (28, 59)
[l1tZDCEtSumsPrinter2] etSums[-1][1] (type, hwPt) = (27, 47)
[l1tZDCEtSumsPrinter2] etSums[0][0] (type, hwPt) = (28, 845)
[l1tZDCEtSumsPrinter2] etSums[0][1] (type, hwPt) = (27, 52)
[l1tZDCEtSumsPrinter2] etSums[1][0] (type, hwPt) = (28, 1)
[l1tZDCEtSumsPrinter2] etSums[1][1] (type, hwPt) = (27, 46)
[l1tZDCEtSumsPrinter2] etSums[2][0] (type, hwPt) = (28, 28)
[l1tZDCEtSumsPrinter2] etSums[2][1] (type, hwPt) = (27, 35)
[l1tZDCEtSumsPrinter2] etSums[-1][0] (type, hwPt) = (28, 19)
[l1tZDCEtSumsPrinter2] etSums[-1][1] (type, hwPt) = (27, 66)
[l1tZDCEtSumsPrinter2] etSums[0][0] (type, hwPt) = (28, 20)
[l1tZDCEtSumsPrinter2] etSums[0][1] (type, hwPt) = (27, 61)
[l1tZDCEtSumsPrinter2] etSums[1][0] (type, hwPt) = (28, 23)
[l1tZDCEtSumsPrinter2] etSums[1][1] (type, hwPt) = (27, 47)
[l1tZDCEtSumsPrinter2] etSums[2][0] (type, hwPt) = (28, 12)
[l1tZDCEtSumsPrinter2] etSums[2][1] (type, hwPt) = (27, 45)
@EOF

# compare to expected output of test job
diff log_testL1TZDCEtSumsProducer_l1tZDCEtSumsPrinter2_expected \
     log_testL1TZDCEtSumsProducer_l1tZDCEtSumsPrinter2 \
 || die "differences in expected log report" $?
