#!/bin/bash -ex

ERR=0
echo ">>> Create temporary directory for cache"
TEST_TMPDIR=$(mktemp -d /tmp/cmssw_theano.XXXXXXX)

echo ">>> Change default behaviour for Theano"
export THEANO_FLAGS="device=cpu,force_device=True,base_compiledir=$TEST_TMPDIR"

echo ">>> Theano configuration for testing:"
python -c 'import theano; print(theano.config)' || ERR=1

echo ">>> Cleaning compile cache"
theano-cache clear || ERR=1

for i in $(cat ${CMSSW_BASE}/src/PhysicsTools/PythonAnalysis/test/imports.txt)
do
   echo "importing $i"
   python -c "import $i" || ERR=1
done

echo ">>> Cleaning compile cache"
theano-cache clear || ERR=1

for i in $(cat ${CMSSW_BASE}/src/PhysicsTools/PythonAnalysis/test/commands.txt)
do
   echo "testing $i"
   $i -h || ERR=1
done

rm -rf "$TEST_TMPDIR"
exit $ERR
