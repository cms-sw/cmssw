#!/bin/sh
LOCAL_TEST_DIR=$(dirname $0)
function die { echo $1: status $2 ;  exit $2; }

python ${LOCAL_TEST_DIR}/testHistogrammar.py || die 'Failure using testHistogrammar' $? 
python ${LOCAL_TEST_DIR}/testPandas.py || die 'Failure using testPandas' $?
python ${LOCAL_TEST_DIR}/testRootNumpy.py || die 'Failure using testRootNumpy' $?
python ${LOCAL_TEST_DIR}/test_pycurl.py || die 'Failure using test_pycurl' $?
#python ${LOCAL_TEST_DIR}/testRootpy.py || die 'Failure using testRootpy' $?
