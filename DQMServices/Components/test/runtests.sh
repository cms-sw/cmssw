#!/bin/sh

function die { echo $1; status $2; exit $2; }
eval `scram runtime -sh`
python ${LOCAL_TEST_DIR}/whiteRabbit.py  -q 1 -j 1 -n 11 --noMail || die 'Failure in running all tests' $?
