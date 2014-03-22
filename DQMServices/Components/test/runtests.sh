#!/bin/sh

function die { echo $1; status $2; exit $2; }
eval `scram runtime -sh`
python ${LOCAL_TEST_DIR}/whiteRabbit.py  -q 1 -j 5 -n 1,2,3,4,6,7,8,9,11,12 --noMail || die 'Failure in running all tests' $?
