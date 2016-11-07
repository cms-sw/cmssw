#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

python ${LOCAL_TEST_DIR}/testHistogrammar.py || die 'Failure using testHistogrammar' $?
