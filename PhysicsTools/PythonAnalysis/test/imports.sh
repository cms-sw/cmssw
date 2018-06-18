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

python ${CMSSW_BASE}/src/PhysicsTools/PythonAnalysis/test/imports.py

echo ">>> Cleaning compile cache"
theano-cache clear

rm -rf "$TEST_TMPDIR"
