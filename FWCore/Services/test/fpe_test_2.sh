#!/bin/bash
# If successful this will dump core so disable that... 
ulimit -c 0

CONFIG=${LOCAL_TEST_DIR}/fpe_test_2_cfg.py

echo "***"
echo "If the test is successful, cmsRun will fail (abort) with error status 136 or 11."
echo "The purpose is to test that floating point exceptions cause failures."
echo "If the floating point exception does not cause the cmsRun job to"
echo "abort, an explicit exception will thrown from CMS code that also causes"
echo "an abort, but this time with error status 65."
echo "The values 136 and 11 depend on things underneath that CMS does not"
echo "control.  These values have changed before and may change again. If"
echo "they do, someone will need to investigate and change the criteria in"
echo "this shell script (fpe_test_2.sh)."
echo "***"


# With all exceptions set to false, cmsRun should complete with status 0
export OVERFLOW=False
export DIVIDEBYZERO=False
export INVALID=False
export UNDERFLOW=False 
#totalview cmsRun -a ${CONFIG}
cmsRun ${CONFIG} >& /dev/null
status=$?
unset OVERLFOW
unset DIVIDEBYZERO
unset INVALID
unset OVERFLOW

echo "Completed cmsRun with FP exceptions disabled"
echo "cmsRun status: " $status
if [ $status -ne 0 ] ; then
 echo "Test FAILED, status not 0"
 exit 1
fi

# DIVIDEBYZERO
export OVERFLOW=False
export DIVIDEBYZERO=True
export INVALID=False
export UNDERFLOW=False 
cmsRun ${CONFIG} >& /dev/null
status=$?
unset OVERLFOW
unset DIVIDEBYZERO
unset INVALID
unset OVERFLOW

echo "Completed cmsRun with DIVIDEBYZERO exception enabled"
echo "cmsRun status: " $status
if [ $status -ne 136 -a $status -ne 11 ] ; then
 echo "Test FAILED, status neither 136 nor 11"
 exit 1
fi

# INVALID
export OVERFLOW=False
export DIVIDEBYZERO=False
export INVALID=True
export UNDERFLOW=False 
cmsRun ${CONFIG} >& /dev/null
status=$?
unset OVERLFOW
unset DIVIDEBYZERO
unset INVALID
unset OVERFLOW

echo "Completed cmsRun with INVALID exception enabled"
echo "cmsRun status: " $status
if [ $status -ne 136 -a $status -ne 11 ] ; then
 echo "Test FAILED, status neither 136 nor 11"
 exit 1
fi

# OVERFLOW
export OVERFLOW=True
export DIVIDEBYZERO=False
export INVALID=False
export UNDERFLOW=False 
cmsRun ${CONFIG} >& /dev/null
status=$?
unset OVERLFOW
unset DIVIDEBYZERO
unset INVALID
unset OVERFLOW

echo "Completed cmsRun with OVERFLOW exception enabled"
echo "cmsRun status: " $status
if [ $status -ne 136 -a $status -ne 11 ] ; then
 echo "Test FAILED, status neither 136 nor 11"
 exit 1
fi

# UNDERFLOW
export OVERFLOW=False
export DIVIDEBYZERO=False
export INVALID=False
export UNDERFLOW=True
cmsRun ${CONFIG} >& /dev/null
status=$?
unset OVERLFOW
unset DIVIDEBYZERO
unset INVALID
unset OVERFLOW

echo "Completed cmsRun with UNDERFLOW exception enabled"
echo "cmsRun status: " $status
if [ $status -ne 136 -a $status -ne 11 ] ; then
 echo "Test FAILED, status neither 136 nor 11"
 exit 1
fi

echo "Test SUCCEEDED"
