#!/bin/bash

set -x
set -e

echo ">>> Create temporary directory for cache"
TEST_TMPDIR=$(mktemp -d /tmp/cmssw_theano.XXXXXXX)

echo ">>> Change default behaviour for Theano"
export THEANO_FLAGS="device=cpu,force_device=True,base_compiledir=$TEST_TMPDIR"

echo ">>> Theano configuration for testing:"
python -c 'import theano; print(theano.config)'

echo ">>> Cleaning compile cache"
theano-cache clear

for i in $(cat ${CMSSW_BASE}/src/PhysicsTools/PythonAnalysis/test/imports.txt)
do
   echo "importing $i"
   python -c "import $i"
done

echo ">>> Cleaning compile cache"
theano-cache clear

for i in $(cat ${CMSSW_BASE}/src/PhysicsTools/PythonAnalysis/test/commands.txt)
do
   echo "testing $i"
   $i -h
done


rm -rf "$TEST_TMPDIR"
