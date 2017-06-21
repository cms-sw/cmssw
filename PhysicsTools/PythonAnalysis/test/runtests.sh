#!/bin/sh
LOCAL_TEST_DIR=$(dirname $0)
function die { echo $1: status $2 ;  exit $2; }

python ${LOCAL_TEST_DIR}/testHistogrammar.py || die 'Failure using testHistogrammar' $? 
python ${LOCAL_TEST_DIR}/testPandas.py || die 'Failure using testPandas' $?
python ${LOCAL_TEST_DIR}/testRootNumpy.py || die 'Failure using testRootNumpy' $?
python ${LOCAL_TEST_DIR}/test_pycurl.py || die 'Failure using test_pycurl' $?
python ${LOCAL_TEST_DIR}/testRootpy.py || die 'Failure using testRootpy' $?

python ${LOCAL_TEST_DIR}/testBottleneck.py || die 'Failure using testBottleneck' $?
python ${LOCAL_TEST_DIR}/testDeepDish.py || die 'Failure using testDeepDish' $?
python ${LOCAL_TEST_DIR}/testNumExpr.py || die 'Failure using testNumExpr' $?
python ${LOCAL_TEST_DIR}/testNumba.py || die 'Failure using testNumba' $?
python ${LOCAL_TEST_DIR}/testTables.py || die 'Failure using testTables' $?

python ${LOCAL_TEST_DIR}/testDownhill.py || die 'Failure using testDownhill' $?
python ${LOCAL_TEST_DIR}/testXGBoost_and_sklearn.py || die 'Failure using testXGBoost' $?
#python ${LOCAL_TEST_DIR}/testTheanets.py || die 'Failure using testTheanets' $?

python ${LOCAL_TEST_DIR}/testhep_ml.py || die 'Failure using testhep_ml' $?
python ${LOCAL_TEST_DIR}/testUncertainties.py || die 'Failure using testUncertainties' $?
