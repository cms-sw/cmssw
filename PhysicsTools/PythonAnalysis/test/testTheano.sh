#!/bin/bash -x

ERR=0
echo ">>> Create temporary directory for cache"
TEST_TMPDIR=$(mktemp -d ${CMSSW_BASE}/tmp/cmssw_theano.XXXXXXX)

echo ">>> Change default behaviour for Theano"
export THEANO_FLAGS="device=cpu,force_device=True,base_compiledir=$TEST_TMPDIR"

echo ">>> Theano configuration for testing:"
python -c 'import theano; print(theano.config)' || ERR=1

echo ">>> Cleaning compile cache"
theano-cache clear  || ERR=1

if [ "$1" != "" ] ; then
  python ${CMSSW_BASE}/src/PhysicsTools/PythonAnalysis/test/$1  || ERR=1
fi

echo ">>> Cleaning compile cache"
theano-cache clear  || ERR=1

rm -rf "$TEST_TMPDIR"
exit $ERR
